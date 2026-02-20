#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    build_carbonpdf_prompt,
    build_system_instruction,
    ensure_dir,
    extract_numeric_prediction,
    is_correct_numeric,
    sanitize_model_name,
    save_json,
    save_jsonl,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot evaluation on CarbonPDF-QA")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--cache_dir", type=str, default="/workspace/.cache/huggingface")
    p.add_argument("--data_path", type=str, default="")
    p.add_argument("--mock_data", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable model internal thinking in chat template (default: disabled).",
    )
    return p.parse_args()


def _load_local_dataset(path: str) -> Dataset:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict):
            rows = obj.get("data") or obj.get("examples") or obj.get("items") or [obj]
        else:
            raise ValueError(f"Unsupported json structure in {path}")

    normalized = []
    for r in rows:
        normalized.append(
            {
                "question": r.get("question", ""),
                "context": r.get("context", r.get("passage", r.get("document", ""))),
                "answer": r.get("answer", r.get("gold", r.get("label"))),
            }
        )
    return Dataset.from_list(normalized)


def _load_mock_dataset() -> Dataset:
    rows = [
        {"question": "What is total emissions in 2023?", "context": "Total emissions were 125.5 tons.", "answer": 125.5},
        {"question": "What is renewable ratio?", "context": "Renewable ratio reached 42%.", "answer": 0.42},
    ]
    return Dataset.from_list(rows)


def init_model(model_name: str, cache_dir: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }

    try:
        model_kwargs["load_in_8bit"] = True
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **model_kwargs)
    except Exception:
        model_kwargs.pop("load_in_8bit", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **model_kwargs)

    return tokenizer, model


def to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    try:
        s = str(x).strip().replace(",", "")
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)
    except Exception:
        return None


def update_summary(summary_path: str, run_record: Dict[str, Any]) -> None:
    summary: Dict[str, Any] = {"runs": []}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            if "runs" not in summary or not isinstance(summary["runs"], list):
                summary = {"runs": []}
        except Exception:
            summary = {"runs": []}

    summary["runs"].append(run_record)
    save_json(summary_path, summary)


def _prepare_model_inputs(tokenizer, model, messages: List[Dict[str, str]], enable_thinking: bool) -> Dict[str, Any]:
    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs = dict(
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        try:
            model_inputs = tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                **template_kwargs,
            )
        except TypeError:
            model_inputs = tokenizer.apply_chat_template(messages, **template_kwargs)
    else:
        text = messages[0]["content"] + "\n\n" + messages[1]["content"]
        model_inputs = tokenizer(text, return_tensors="pt")

    if isinstance(model_inputs, torch.Tensor):
        model_inputs = {"input_ids": model_inputs}
    elif hasattr(model_inputs, "data"):
        model_inputs = dict(model_inputs)
    elif not isinstance(model_inputs, dict):
        raise TypeError(f"Unsupported model input type: {type(model_inputs)}")

    return {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(args.results_dir)
    ensure_dir(args.cache_dir)
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.cache_dir, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.cache_dir, "transformers")
    ensure_dir(os.environ["HUGGINGFACE_HUB_CACHE"])
    ensure_dir(os.environ["TRANSFORMERS_CACHE"])

    if args.data_path:
        ds = _load_local_dataset(args.data_path)
    elif args.mock_data:
        ds = _load_mock_dataset()
    else:
        raise RuntimeError("Provide --data_path or use --mock_data")

    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    tokenizer, model = init_model(args.model_name, cache_dir=args.cache_dir, trust_remote_code=args.trust_remote_code)
    system_prompt = build_system_instruction()

    rows: List[Dict[str, Any]] = []
    n_total = 0
    n_parse_fail = 0
    n_correct = 0

    for ex in tqdm(ds, total=len(ds), desc="Evaluating CarbonPDF-QA"):
        question = str(ex.get("question", ""))
        gold = to_float(ex.get("answer"))
        prompt = build_carbonpdf_prompt(ex)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        model_inputs = _prepare_model_inputs(
            tokenizer,
            model,
            messages,
            enable_thinking=args.enable_thinking,
        )
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0][input_len:]
        raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred = extract_numeric_prediction(raw_output)

        parse_fail = pred is None
        correct = is_correct_numeric(pred, gold, atol=args.atol, rtol=args.rtol)

        if parse_fail:
            n_parse_fail += 1
        if correct:
            n_correct += 1

        rows.append(
            {
                "question": question,
                "gold": gold,
                "pred": pred,
                "raw_output": raw_output,
                "correct": correct,
                "parse_fail": parse_fail,
            }
        )
        n_total += 1

    accuracy = (n_correct / n_total) if n_total else 0.0
    parse_fail_rate = (n_parse_fail / n_total) if n_total else 0.0

    safe_model = sanitize_model_name(args.model_name)
    jsonl_path = os.path.join(args.results_dir, f"carbonpdf_{safe_model}_{args.split}.jsonl")
    save_jsonl(jsonl_path, rows)

    run_record = {
        "task": "carbonpdf",
        "model": args.model_name,
        "split": args.split,
        "num_samples": n_total,
        "max_new_tokens": args.max_new_tokens,
        "atol": args.atol,
        "rtol": args.rtol,
        "accuracy": accuracy,
        "parse_fail_rate": parse_fail_rate,
        "jsonl": jsonl_path,
        "data_path": args.data_path or "mock_data",
    }

    summary_path = os.path.join(args.results_dir, "summary.json")
    update_summary(summary_path, run_record)

    print(json.dumps(run_record, ensure_ascii=False, indent=2))
    print(f"[saved] {jsonl_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
