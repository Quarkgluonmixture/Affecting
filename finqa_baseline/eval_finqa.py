#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    build_error_cases_markdown,
    build_finqa_prompt,
    build_system_instruction,
    ensure_dir,
    extract_numeric_prediction,
    is_correct_numeric,
    normalize_gold_numeric,
    sanitize_model_name,
    save_json,
    save_jsonl,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot evaluation on FinQA")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--setting", type=str, choices=["oracle", "full"], default="oracle")
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--cache_dir", type=str, default="/workspace/.cache/huggingface")
    p.add_argument("--dataset_name", type=str, default="", help="Optional HF dataset name override")
    p.add_argument("--dataset_config", type=str, default="", help="Optional HF dataset config")
    p.add_argument("--dataset_split", type=str, default="", help="Optional HF split override")
    p.add_argument("--local_json", type=str, default="", help="Optional local json/jsonl file for FinQA")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable model internal thinking in chat template (default: disabled).",
    )
    p.add_argument(
        "--percent_auto_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When question is percentage-like and gold<1<pred, also evaluate pred/100.",
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
    return Dataset.from_list(rows)


def load_finqa_dataset(args: argparse.Namespace) -> Dataset:
    if args.local_json:
        return _load_local_dataset(args.local_json)

    split = args.dataset_split or args.split

    if args.dataset_name:
        cfg = args.dataset_config if args.dataset_config else None
        return load_dataset(
            args.dataset_name,
            cfg,
            split=split,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )

    candidates: List[Tuple[str, Optional[str]]] = [
        ("dreamerdeo/finqa", None),
        ("ibm-research/finqa", None),
        ("finqa", None),
        ("ibm/finqa", None),
        ("yale-nlp/finqa", None),
    ]

    last_err: Optional[Exception] = None
    for name, cfg in candidates:
        try:
            ds = load_dataset(
                name,
                cfg,
                split=split,
                cache_dir=args.cache_dir,
                trust_remote_code=args.trust_remote_code,
            )
            print(f"[data] loaded FinQA from {name} split={split}")
            return ds
        except Exception as e:
            last_err = e

    hint = ""
    if last_err is not None and "Dataset scripts are no longer supported" in str(last_err):
        hint = " Detected datasets>=4 script restriction; install `datasets<4`."

    raise RuntimeError(
        "Failed to load FinQA from HF. Use --dataset_name/--dataset_config or --local_json. "
        f"Last error: {last_err}.{hint}"
    )


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


def format_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


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


def generate_one(
    tokenizer,
    model,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    model_inputs = _prepare_model_inputs(tokenizer, model, messages, enable_thinking=enable_thinking)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _is_percent_question(question: str) -> bool:
    q = question.lower()
    return ("percent" in q) or ("percentage" in q) or ("%" in q)


def _evaluate_with_optional_percent_autoscale(
    pred: Optional[float],
    gold: Optional[float],
    question: str,
    atol: float,
    rtol: float,
    percent_auto_scale: bool,
) -> Dict[str, Any]:
    base_correct = is_correct_numeric(pred, gold, atol=atol, rtol=rtol)
    adjusted_correct = base_correct
    percent_recovered = False
    adjusted_pred = pred

    if (
        percent_auto_scale
        and (not base_correct)
        and pred is not None
        and gold is not None
        and _is_percent_question(question)
        and abs(float(gold)) < 1.0
        and abs(float(pred)) > 1.0
    ):
        scaled = float(pred) / 100.0
        if is_correct_numeric(scaled, gold, atol=atol, rtol=rtol):
            adjusted_correct = True
            percent_recovered = True
            adjusted_pred = scaled

    return {
        "base_correct": base_correct,
        "adjusted_correct": adjusted_correct,
        "percent_recovered": percent_recovered,
        "adjusted_pred": adjusted_pred,
    }


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

    ds = load_finqa_dataset(args)
    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    tokenizer, model = init_model(args.model_name, cache_dir=args.cache_dir, trust_remote_code=args.trust_remote_code)
    system_prompt = build_system_instruction()

    rows: List[Dict[str, Any]] = []
    n_total = 0
    n_parse_fail = 0
    n_correct_base = 0
    n_correct_adjusted = 0
    n_percent_recovered = 0

    for ex in tqdm(ds, total=len(ds), desc="Evaluating FinQA"):
        question = str(ex.get("question", ""))
        gold = normalize_gold_numeric(ex)
        prompt = build_finqa_prompt(ex, setting=args.setting)
        messages = format_messages(system_prompt, prompt)

        raw_output = generate_one(
            tokenizer,
            model,
            messages,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
        )
        pred = extract_numeric_prediction(raw_output)

        parse_fail = pred is None
        eval_flags = _evaluate_with_optional_percent_autoscale(
            pred=pred,
            gold=gold,
            question=question,
            atol=args.atol,
            rtol=args.rtol,
            percent_auto_scale=args.percent_auto_scale,
        )
        base_correct = bool(eval_flags["base_correct"])
        adjusted_correct = bool(eval_flags["adjusted_correct"])
        percent_recovered = bool(eval_flags["percent_recovered"])
        adjusted_pred = eval_flags["adjusted_pred"]

        if parse_fail:
            n_parse_fail += 1
        if base_correct:
            n_correct_base += 1
        if adjusted_correct:
            n_correct_adjusted += 1
        if percent_recovered:
            n_percent_recovered += 1

        rows.append(
            {
                "question": question,
                "gold": gold,
                "pred": pred,
                "adjusted_pred": adjusted_pred,
                "raw_output": raw_output,
                "correct_base": base_correct,
                "correct": adjusted_correct,
                "percent_recovered": percent_recovered,
                "parse_fail": parse_fail,
            }
        )
        n_total += 1

    accuracy_base = (n_correct_base / n_total) if n_total else 0.0
    accuracy_adjusted = (n_correct_adjusted / n_total) if n_total else 0.0
    parse_fail_rate = (n_parse_fail / n_total) if n_total else 0.0

    safe_model = sanitize_model_name(args.model_name)
    jsonl_path = os.path.join(args.results_dir, f"finqa_{safe_model}_{args.setting}_{args.split}.jsonl")
    save_jsonl(jsonl_path, rows)

    error_cases_path = os.path.join(args.results_dir, "error_cases.md")
    with open(error_cases_path, "w", encoding="utf-8") as f:
        f.write(build_error_cases_markdown(rows, max_cases=20))

    run_record = {
        "task": "finqa",
        "model": args.model_name,
        "split": args.split,
        "setting": args.setting,
        "num_samples": n_total,
        "max_new_tokens": args.max_new_tokens,
        "atol": args.atol,
        "rtol": args.rtol,
        "percent_auto_scale": args.percent_auto_scale,
        "accuracy": accuracy_adjusted,
        "accuracy_base": accuracy_base,
        "accuracy_adjusted": accuracy_adjusted,
        "percent_recovered_count": n_percent_recovered,
        "parse_fail_rate": parse_fail_rate,
        "jsonl": jsonl_path,
    }

    summary_path = os.path.join(args.results_dir, "summary.json")
    update_summary(summary_path, run_record)

    print(json.dumps(run_record, ensure_ascii=False, indent=2))
    print(f"[saved] {jsonl_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {error_cases_path}")


if __name__ == "__main__":
    main()
