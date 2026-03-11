#!/usr/bin/env python3
"""Generate final report from summary.json for FinQA runs."""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_summary(results_dir: str = "results") -> Dict[str, Any]:
    summary_path = Path(results_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_latest_runs(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs_by_key: Dict[Any, Dict[str, Any]] = {}
    for run in summary.get("runs", []):
        if run.get("task") != "finqa":
            continue
        key = (run.get("model"), run.get("setting"), run.get("split"))
        runs_by_key[key] = run
    return list(runs_by_key.values())


def _sort_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(run: Dict[str, Any]):
        model = str(run.get("model", ""))
        setting = str(run.get("setting", ""))
        model_order = 0 if "4B" in model else 1
        setting_order = 0 if setting == "oracle" else 1
        return (model_order, setting_order, model)

    return sorted(runs, key=sort_key)


def _safe_rate(d: Dict[str, Any], key: str, fallback: float = 0.0) -> float:
    v = d.get(key, fallback)
    try:
        return float(v)
    except Exception:
        return fallback


def _get_tag_open_rate(run: Dict[str, Any]) -> float:
    counts = run.get("tag_status_counts", {})
    if not isinstance(counts, dict):
        return 0.0
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return 0.0
    return float(counts.get("open_only", 0)) / float(total)


def generate_report(summary: Dict[str, Any], output_path: str = "results/final_report.md") -> None:
    runs = _sort_runs(get_latest_runs(summary))

    md_lines = [
        "# Final Baseline Report (FinQA, Zero-Shot)",
        "",
        "## Scope",
        "- Dataset: FinQA test split",
        "- Decoding: greedy (`do_sample=False`)",
        "- Default baseline: thinking=true + `[FINAL_ANSWER]...[/FINAL_ANSWER]` + `math_verify`",
        "- Also reported: legacy numeric evaluator for side-by-side comparison",
        "",
        "## Baseline Table (Latest per Model/Setting)",
        "| model | setting | split | n | acc_mathverify | acc_legacy | delta | parse_fail_mv | parse_fail_legacy | tag_open_only_rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    best_acc = -1.0
    best_run: Dict[str, Any] = {}

    for run in runs:
        model = run.get("model", "")
        setting = run.get("setting", "")
        split = run.get("split", "")
        n = int(run.get("num_samples", 0))

        acc_mv = _safe_rate(run, "accuracy_mathverify", _safe_rate(run, "accuracy"))
        acc_legacy = _safe_rate(run, "accuracy_legacy", _safe_rate(run, "accuracy_adjusted"))
        pf_mv = _safe_rate(run, "parse_fail_rate_mathverify", _safe_rate(run, "parse_fail_rate"))
        pf_legacy = _safe_rate(run, "parse_fail_rate_legacy")
        open_rate = _get_tag_open_rate(run)

        md_lines.append(
            f"| {model} | {setting} | {split} | {n} | {acc_mv:.6f} | {acc_legacy:.6f} | {(acc_mv - acc_legacy):.6f} | {pf_mv:.6f} | {pf_legacy:.6f} | {open_rate:.6f} |"
        )

        if acc_mv > best_acc:
            best_acc = acc_mv
            best_run = run

    md_lines.extend([
        "",
        "## Key Findings",
    ])

    if best_run:
        md_lines.append(
            f"- Best math-verify run: {best_run.get('model')} / {best_run.get('setting')} @ {best_acc:.4f}"
        )

    if runs:
        avg_pf_mv = sum(_safe_rate(r, "parse_fail_rate_mathverify", _safe_rate(r, "parse_fail_rate")) for r in runs) / len(runs)
        avg_pf_legacy = sum(_safe_rate(r, "parse_fail_rate_legacy") for r in runs) / len(runs)
        md_lines.append(f"- Avg parse_fail (math-verify): {avg_pf_mv:.4f}")
        md_lines.append(f"- Avg parse_fail (legacy): {avg_pf_legacy:.4f}")

    md_lines.extend(
        [
            "- Tag-status distribution is recorded in `summary.json` (`tag_status_counts`) for truncation diagnostics.",
            "- `open_only` tags indicate truncated close tags; extraction fallback still provides answer text.",
            "",
            "## Artifacts",
            "- `results/regression_final_answer_mathverify.md`",
            "- `results/summary.json`",
            "- `results/error_cases.md`",
        ]
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Report generated: {out}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate final report from evaluation results")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--output", type=str, default="results/final_report.md", help="Output report path")
    args = parser.parse_args()

    summary = load_summary(args.results_dir)
    generate_report(summary, args.output)


if __name__ == "__main__":
    main()
