#!/usr/bin/env python3
"""
Build audited training targets for answer_only / formula_rationale supervision.

Key behaviors:
- execute `program` to derive formula_value when possible
- compare gold_value vs formula_value using scale relation tags
- correct training answer target to formula_value whenever relation != consistent
- emit audit fields for downstream analysis
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List
import sys

CURRENT_DIR = Path(__file__).resolve().parent
STAGE1_ROOT = CURRENT_DIR.parent
if str(STAGE1_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE1_ROOT))

from src.preprocessing.formula_utils import derive_formula_metadata, format_number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build formula-rationale targets with percent-scale consistency audit."
    )
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument(
        "--summary_json",
        type=str,
        default="",
        help="Optional summary output path.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)

    rows = load_jsonl(input_path)

    scale_counter: Counter[str] = Counter()
    bucket_counter: Counter[str] = Counter()
    corrected_count = 0
    exec_ok_count = 0
    sample_corrections: List[Dict[str, Any]] = []

    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        meta = derive_formula_metadata(row)
        scale_counter.update([str(meta["scale_relation"])])
        bucket_counter.update([str(meta["program_steps_bucket"])])

        if bool(meta["formula_exec_ok"]):
            exec_ok_count += 1
        if bool(meta["answer_corrected"]):
            corrected_count += 1
            if len(sample_corrections) < 20:
                sample_corrections.append(
                    {
                        "id": row.get("id", ""),
                        "gold_raw": meta["gold_raw"],
                        "gold_value": meta["gold_value"],
                        "formula_value": meta["formula_value"],
                        "scale_relation": meta["scale_relation"],
                        "formula_expression": meta["formula_expression"],
                        "corrected_answer": meta["corrected_answer"],
                    }
                )

        enriched = dict(row)
        enriched.update(meta)
        # target_answer is the canonical answer used by training prompts.
        enriched["target_answer"] = str(meta["corrected_answer"])
        if meta["formula_value"] is not None:
            enriched["formula_value_text"] = format_number(float(meta["formula_value"]))
        else:
            enriched["formula_value_text"] = ""

        out_rows.append(enriched)

    save_jsonl(output_path, out_rows)

    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "num_rows": len(rows),
        "formula_exec_ok_count": exec_ok_count,
        "formula_exec_ok_rate": (exec_ok_count / len(rows)) if rows else 0.0,
        "answer_corrected_count": corrected_count,
        "answer_corrected_rate": (corrected_count / len(rows)) if rows else 0.0,
        "scale_relation_counts": dict(scale_counter),
        "program_steps_bucket_counts": dict(bucket_counter),
        "sample_corrections": sample_corrections,
    }

    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    save_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
