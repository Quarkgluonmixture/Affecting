import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def classify_error_case(row: Dict[str, Any]) -> str:
    if row.get("error_type"):
        return str(row["error_type"])

    raw = str(row.get("raw_output", "")).lower()
    gold = row.get("gold")
    pred = row.get("pred")

    if row.get("pred") is None:
        return "parse fail"

    if "%" in raw:
        return "percent scaling"

    unit_hints = ["million", "billion", "thousand", "k", "m", "bn", "usd", "dollar", "kg", "ton"]
    if any(h in raw for h in unit_hints):
        return "unit confusion"

    if pred is not None and gold is not None and abs(float(pred) - float(gold)) > 1e-3:
        return "numeric mismatch"

    return "numeric mismatch"


def build_error_cases_markdown(rows: List[Dict[str, Any]], max_cases: int = 20) -> str:
    wrong = [r for r in rows if not r.get("correct", False)]
    buckets = defaultdict(list)
    for r in wrong:
        buckets[classify_error_case(r)].append(r)

    order = ["parse fail", "numeric mismatch", "percent scaling", "unit confusion"]
    selected: List[Dict[str, Any]] = []

    for k in order:
        selected.extend(buckets.get(k, [])[: max(1, max_cases // len(order))])
        if len(selected) >= max_cases:
            break

    if len(selected) < max_cases:
        for k, vals in buckets.items():
            if k in order:
                continue
            for v in vals:
                selected.append(v)
                if len(selected) >= max_cases:
                    break
            if len(selected) >= max_cases:
                break

    lines = ["# Error Cases (Auto Extracted)", ""]
    if not selected:
        lines.append("No failed samples found.")
        return "\n".join(lines)

    for i, r in enumerate(selected[:max_cases], start=1):
        lines.extend(
            [
                f"## {i}. {classify_error_case(r)}",
                f"- question: {r.get('question', '')}",
                f"- gold: {r.get('gold', None)}",
                f"- pred: {r.get('pred', None)}",
                f"- raw_output: {r.get('raw_output', '')}",
                "",
            ]
        )

    return "\n".join(lines)
