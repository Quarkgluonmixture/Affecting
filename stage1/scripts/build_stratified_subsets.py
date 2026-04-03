#!/usr/bin/env python3
"""
Create nested stratified subsets for Stage-B data ablation.

Guarantees:
- nested subsets: n1 subset is prefix of n2 subset when n1 < n2
- stratification by program-step bucket (single/double/multi)
- reproducible via fixed seed
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

CURRENT_DIR = Path(__file__).resolve().parent
STAGE1_ROOT = CURRENT_DIR.parent
if str(STAGE1_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE1_ROOT))

from src.preprocessing.formula_utils import derive_formula_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build nested stratified subsets from jsonl.")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 1000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id_field", type=str, default="id")
    parser.add_argument("--prefix", type=str, default="train")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _bucket_of(row: Dict[str, Any]) -> str:
    bucket = str(row.get("program_steps_bucket", "")).strip()
    if bucket in {"single", "double", "multi"}:
        return bucket
    meta = derive_formula_metadata(row)
    return str(meta["program_steps_bucket"])


def _stratified_master_order(rows: List[Dict[str, Any]], seed: int) -> List[int]:
    by_bucket: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_bucket[_bucket_of(row)].append(idx)

    rng = random.Random(seed)
    for bucket in list(by_bucket.keys()):
        rng.shuffle(by_bucket[bucket])

    total = len(rows)
    if total == 0:
        return []

    keys = sorted(by_bucket.keys())
    target_ratio: Dict[str, float] = {
        k: (len(by_bucket[k]) / total) for k in keys
    }
    produced: Dict[str, int] = {k: 0 for k in keys}
    consumed: Dict[str, int] = {k: 0 for k in keys}

    order: List[int] = []
    for pos in range(total):
        candidates: List[Tuple[float, float, str]] = []
        for k in keys:
            if consumed[k] >= len(by_bucket[k]):
                continue
            desired_so_far = (pos + 1) * target_ratio[k]
            deficit = desired_so_far - produced[k]
            remaining_ratio = (len(by_bucket[k]) - consumed[k]) / max(1, total - pos)
            candidates.append((deficit, remaining_ratio, k))

        if not candidates:
            break
        # Greedy by current deficit, then by larger remaining ratio.
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        chosen = candidates[0][2]

        local_idx = consumed[chosen]
        order.append(by_bucket[chosen][local_idx])
        consumed[chosen] += 1
        produced[chosen] += 1

    return order


def _subset_stats(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for row in rows:
        c[_bucket_of(row)] += 1
    return dict(c)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_ids(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sid in ids:
            f.write(f"{sid}\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    master_order = _stratified_master_order(rows, seed=int(args.seed))
    ordered_rows = [rows[idx] for idx in master_order]

    sizes = sorted({int(s) for s in args.sizes if int(s) > 0})
    if not sizes:
        raise ValueError("No positive sizes provided.")
    if sizes[-1] > len(rows):
        raise ValueError(f"Largest size {sizes[-1]} exceeds dataset size {len(rows)}")

    summary: Dict[str, Any] = {
        "input_jsonl": str(input_path),
        "seed": int(args.seed),
        "total_rows": len(rows),
        "full_bucket_counts": _subset_stats(rows),
        "subsets": {},
    }

    id_field = str(args.id_field)
    prefix = str(args.prefix)
    for n in sizes:
        subset = ordered_rows[:n]
        subset_ids = [str(row.get(id_field, f"row_{i}")) for i, row in enumerate(subset)]
        subset_jsonl = output_dir / f"{prefix}_{n}.jsonl"
        subset_ids_file = output_dir / f"{prefix}_{n}_ids.txt"
        save_jsonl(subset_jsonl, subset)
        save_ids(subset_ids_file, subset_ids)

        summary["subsets"][str(n)] = {
            "size": n,
            "jsonl": str(subset_jsonl),
            "ids_file": str(subset_ids_file),
            "bucket_counts": _subset_stats(subset),
        }

    summary_path = output_dir / f"{prefix}_subset_summary.json"
    save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
