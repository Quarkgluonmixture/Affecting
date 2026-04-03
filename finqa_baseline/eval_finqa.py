#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
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
    ensure_mathverify_installed,
    evaluate_mathverify,
    extract_final_answer_text,
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
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--cache_dir", type=str, default="/workspace/.cache/huggingface")
    p.add_argument("--dataset_name", type=str, default="", help="Optional HF dataset name override")
    p.add_argument("--dataset_config", type=str, default="", help="Optional HF dataset config")
    p.add_argument("--dataset_split", type=str, default="", help="Optional HF split override")
    p.add_argument("--local_json", type=str, default="", help="Optional local json/jsonl file for FinQA")
    p.add_argument(
        "--adapter_path",
        type=str,
        default="",
        help="Optional LoRA adapter checkpoint path. Base model is still provided via --model_name.",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable model internal thinking in chat template (default: enabled).",
    )
    p.add_argument(
        "--evaluator",
        type=str,
        choices=["math_verify", "numeric_legacy"],
        default="math_verify",
        help="Primary evaluator for reported accuracy/parse fail rate.",
    )
    p.add_argument(
        "--answer_format",
        type=str,
        choices=["final_answer_tag", "plain_numeric"],
        default="final_answer_tag",
        help="Model output format policy.",
    )
    p.add_argument(
        "--final_answer_tag",
        type=str,
        default="FINAL_ANSWER",
        help="Tag used when answer_format=final_answer_tag.",
    )
    p.add_argument(
        "--percent_auto_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When question is percentage-like and gold<1<pred, also evaluate pred/100 for legacy evaluator.",
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


def _pick_tokenizer_source(model_name: str, adapter_path: str) -> str:
    if not adapter_path:
        return model_name

    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        return model_name

    tokenizer_markers = {
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    }
    if any((adapter_dir / marker).exists() for marker in tokenizer_markers):
        return str(adapter_dir)
    return model_name


def init_model(
    model_name: str,
    cache_dir: str,
    trust_remote_code: bool = False,
    adapter_path: str = "",
):
    tokenizer_source = _pick_tokenizer_source(model_name=model_name, adapter_path=adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }

    try:
        model_kwargs["load_in_8bit"] = True
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **model_kwargs)
    except Exception:
        model_kwargs.pop("load_in_8bit", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **model_kwargs)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    return tokenizer, model, tokenizer_source


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
            try:
                model_inputs = tokenizer.apply_chat_template(messages, **template_kwargs)
            except ValueError as e:
                if "chat_template is not set" not in str(e):
                    raise
                text = messages[0]["content"] + "\n\n" + messages[1]["content"]
                model_inputs = tokenizer(text, return_tensors="pt")
        except ValueError as e:
            if "chat_template is not set" not in str(e):
                raise
            text = messages[0]["content"] + "\n\n" + messages[1]["content"]
            model_inputs = tokenizer(text, return_tensors="pt")
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


def _resolve_gold_text(example: dict, gold_numeric: Optional[float]) -> str:
    for key in ["answer", "ans", "gold", "label", "target", "exe_ans"]:
        if key in example:
            val = example.get(key)
            s = str(val).strip() if val is not None else ""
            if s:
                return s

    qa = example.get("qa")
    if isinstance(qa, dict):
        for key in ["answer", "exe_ans", "ans"]:
            val = qa.get(key)
            s = str(val).strip() if val is not None else ""
            if s:
                return s

    if gold_numeric is not None:
        return str(gold_numeric)

    return ""


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ensure_mathverify_installed()

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

    safe_model = sanitize_model_name(args.model_name)
    jsonl_path = os.path.join(args.results_dir, f"finqa_{safe_model}_{args.setting}_{args.split}.jsonl")

    # --- checkpoint: load previously completed rows ---
    rows: List[Dict[str, Any]] = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as _ckpt_f:
            for _line in _ckpt_f:
                _line = _line.strip()
                if _line:
                    rows.append(json.loads(_line))
        if rows:
            print(f"[checkpoint] Resuming: {len(rows)}/{len(ds)} samples already done, skipping.")

    n_skip = len(rows)

    n_total = 0
    n_correct_main = 0
    n_parse_fail_main = 0
    n_correct_mathverify = 0
    n_parse_fail_mathverify = 0
    n_correct_legacy = 0
    n_correct_legacy_base = 0
    n_parse_fail_legacy = 0
    n_percent_recovered = 0
    tag_status_counts: Dict[str, int] = {"closed": 0, "open_only": 0, "absent": 0}

    # recompute counters from checkpoint rows
    for _row in rows:
        n_total += 1
        if _row.get("correct"):
            n_correct_main += 1
        if _row.get("parse_fail"):
            n_parse_fail_main += 1
        if _row.get("correct_mathverify"):
            n_correct_mathverify += 1
        if _row.get("parse_fail_mathverify"):
            n_parse_fail_mathverify += 1
        if _row.get("correct_legacy"):
            n_correct_legacy += 1
        if _row.get("correct_legacy_base"):
            n_correct_legacy_base += 1
        if _row.get("parse_fail_legacy"):
            n_parse_fail_legacy += 1
        if _row.get("percent_recovered"):
            n_percent_recovered += 1
        _ts = _row.get("tag_status", "absent")
        tag_status_counts[_ts] = tag_status_counts.get(_ts, 0) + 1

    tokenizer, model, tokenizer_source = init_model(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        adapter_path=args.adapter_path,
    )
    system_prompt = build_system_instruction(
        answer_format=args.answer_format,
        final_answer_tag=args.final_answer_tag,
    )

    ds_remaining = ds.select(range(n_skip, len(ds))) if n_skip > 0 else ds

    _jsonl_out = open(jsonl_path, "a", encoding="utf-8")

    for ex in tqdm(ds_remaining, total=len(ds), initial=n_skip, desc="Evaluating FinQA"):
        question = str(ex.get("question", ""))
        gold = normalize_gold_numeric(ex)
        gold_text = _resolve_gold_text(ex, gold)
        prompt = build_finqa_prompt(
            ex,
            setting=args.setting,
            answer_format=args.answer_format,
            final_answer_tag=args.final_answer_tag,
        )
        messages = format_messages(system_prompt, prompt)

        raw_output = generate_one(
            tokenizer,
            model,
            messages,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
        )

        extraction = extract_final_answer_text(
            raw_output=raw_output,
            answer_format=args.answer_format,
            final_answer_tag=args.final_answer_tag,
        )
        final_answer_text = extraction["final_answer_text"]
        tag_status = extraction["tag_status"]
        if tag_status not in tag_status_counts:
            tag_status_counts[tag_status] = 0
        tag_status_counts[tag_status] += 1

        # Legacy numeric comparator on extracted final answer text.
        legacy_pred = extract_numeric_prediction(final_answer_text)
        parse_fail_legacy = legacy_pred is None
        legacy_eval = _evaluate_with_optional_percent_autoscale(
            pred=legacy_pred,
            gold=gold,
            question=question,
            atol=args.atol,
            rtol=args.rtol,
            percent_auto_scale=args.percent_auto_scale,
        )
        correct_legacy_base = bool(legacy_eval["base_correct"])
        correct_legacy = bool(legacy_eval["adjusted_correct"])
        percent_recovered = bool(legacy_eval["percent_recovered"])
        adjusted_legacy_pred = legacy_eval["adjusted_pred"]

        if parse_fail_legacy:
            n_parse_fail_legacy += 1
        if correct_legacy_base:
            n_correct_legacy_base += 1
        if correct_legacy:
            n_correct_legacy += 1
        if percent_recovered:
            n_percent_recovered += 1

        mv_eval = evaluate_mathverify(gold_text=gold_text, pred_text=final_answer_text)
        correct_mathverify = bool(mv_eval["correct"])
        parse_fail_mathverify = bool(mv_eval["parse_fail"])
        mathverify_error = str(mv_eval.get("error", ""))

        if parse_fail_mathverify:
            n_parse_fail_mathverify += 1
        if correct_mathverify:
            n_correct_mathverify += 1

        if args.evaluator == "math_verify":
            correct_main = correct_mathverify
            parse_fail_main = parse_fail_mathverify
        else:
            correct_main = correct_legacy
            parse_fail_main = parse_fail_legacy

        if parse_fail_main:
            n_parse_fail_main += 1
        if correct_main:
            n_correct_main += 1

        _row = {
            "question": question,
            "gold": gold,
            "gold_text": gold_text,
            "raw_output": raw_output,
            "final_answer_text": final_answer_text,
            "tag_status": tag_status,
            "pred": adjusted_legacy_pred,
            "pred_legacy_raw": legacy_pred,
            "pred_legacy_adjusted": adjusted_legacy_pred,
            "correct": correct_main,
            "correct_mathverify": correct_mathverify,
            "correct_legacy": correct_legacy,
            "correct_legacy_base": correct_legacy_base,
            "parse_fail": parse_fail_main,
            "parse_fail_mathverify": parse_fail_mathverify,
            "parse_fail_legacy": parse_fail_legacy,
            "percent_recovered": percent_recovered,
            "mathverify_error": mathverify_error,
            "evaluator": args.evaluator,
        }
        rows.append(_row)
        _jsonl_out.write(json.dumps(_row, ensure_ascii=False) + "\n")
        _jsonl_out.flush()
        n_total += 1

    _jsonl_out.close()

    accuracy_main = (n_correct_main / n_total) if n_total else 0.0
    parse_fail_rate_main = (n_parse_fail_main / n_total) if n_total else 0.0

    accuracy_mathverify = (n_correct_mathverify / n_total) if n_total else 0.0
    parse_fail_rate_mathverify = (n_parse_fail_mathverify / n_total) if n_total else 0.0

    accuracy_legacy = (n_correct_legacy / n_total) if n_total else 0.0
    accuracy_legacy_base = (n_correct_legacy_base / n_total) if n_total else 0.0
    parse_fail_rate_legacy = (n_parse_fail_legacy / n_total) if n_total else 0.0

    error_cases_path = os.path.join(args.results_dir, "error_cases.md")
    with open(error_cases_path, "w", encoding="utf-8") as f:
        f.write(build_error_cases_markdown(rows, max_cases=20))

    run_record = {
        "task": "finqa",
        "model": args.model_name,
        "adapter_path": args.adapter_path,
        "tokenizer_source": tokenizer_source,
        "split": args.split,
        "setting": args.setting,
        "num_samples": n_total,
        "max_new_tokens": args.max_new_tokens,
        "atol": args.atol,
        "rtol": args.rtol,
        "enable_thinking": args.enable_thinking,
        "evaluator": args.evaluator,
        "answer_format": args.answer_format,
        "final_answer_tag": args.final_answer_tag,
        "percent_auto_scale": args.percent_auto_scale,
        "accuracy": accuracy_main,
        "parse_fail_rate": parse_fail_rate_main,
        "accuracy_mathverify": accuracy_mathverify,
        "accuracy_legacy": accuracy_legacy,
        "accuracy_legacy_base": accuracy_legacy_base,
        "accuracy_base": accuracy_legacy_base,
        "accuracy_adjusted": accuracy_legacy,
        "parse_fail_rate_mathverify": parse_fail_rate_mathverify,
        "parse_fail_rate_legacy": parse_fail_rate_legacy,
        "parse_fail_mathverify": n_parse_fail_mathverify,
        "parse_fail_legacy": n_parse_fail_legacy,
        "percent_recovered_count": n_percent_recovered,
        "tag_status_counts": tag_status_counts,
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
