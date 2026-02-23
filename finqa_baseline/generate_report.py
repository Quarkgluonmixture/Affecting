#!/usr/bin/env python3
"""
Generate final report from summary.json and error analysis files.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_summary(results_dir: str = "results") -> Dict[str, Any]:
    """Load summary.json file."""
    summary_path = Path(results_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    with open(summary_path, "r") as f:
        return json.load(f)


def load_error_breakdown(results_dir: str = "results") -> Dict[str, int]:
    """Load error breakdown from summary_latest_adjusted.json if exists."""
    error_path = Path(results_dir) / "summary_latest_adjusted.json"
    if not error_path.exists():
        return {}
    
    with open(error_path, "r") as f:
        data = json.load(f)
        return data.get("error_breakdown", {})


def get_latest_runs(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get the latest run for each model/setting combination."""
    runs_by_key = {}
    
    for run in summary["runs"]:
        key = (run["model"], run["setting"], run["split"])
        if key not in runs_by_key:
            runs_by_key[key] = run
        else:
            # Keep the later run (assuming runs are in chronological order)
            runs_by_key[key] = run
    
    return list(runs_by_key.values())


def calculate_adjusted_accuracy(run: Dict[str, Any], error_breakdown: Dict[str, int]) -> float:
    """
    Calculate adjusted accuracy with percent auto-scale.
    This is a simplified calculation - in practice, you'd need to recalculate
    from the raw predictions to get the exact adjusted accuracy.
    """
    # If the run already has adjusted accuracy, use it
    if "accuracy_adjusted" in run:
        return run["accuracy_adjusted"]
    
    # Otherwise, estimate based on error breakdown (simplified)
    # In real implementation, you'd recalculate from jsonl file
    base_acc = run["accuracy"]
    # This is a placeholder - actual calculation needs raw predictions
    return base_acc  # Return base accuracy as fallback


def generate_report(summary: Dict[str, Any], error_breakdown: Dict[str, int], output_path: str = "results/final_report.md"):
    """Generate final markdown report."""
    runs = get_latest_runs(summary)
    
    # Sort runs: 4B before 8B, oracle before full
    def sort_key(run):
        model = run["model"]
        setting = run["setting"]
        model_order = 0 if "4B" in model else 1
        setting_order = 0 if setting == "oracle" else 1
        return (model_order, setting_order)
    
    runs.sort(key=sort_key)
    
    md_lines = [
        "# Final Baseline Report (FinQA, Zero-Shot)",
        "",
        "## Scope",
        "- Dataset: FinQA test split",
        "- Metrics: base accuracy, adjusted accuracy (percent auto-scale), parse fail rate",
        "- Decoding: greedy (`do_sample=False`)",
        "- Prompt policy: numeric-only answer; thinking disabled by default",
        "",
        "## Baseline Table (Latest per Model/Setting)",
        "| model | setting | split | n | acc_base | acc_adjusted | delta | parse_fail_rate | recovered |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    
    best_acc = 0
    worst_acc = 1
    best_run = None
    worst_run = None
    
    for run in runs:
        model = run["model"]
        setting = run["setting"]
        split = run["split"]
        n = run["num_samples"]
        acc_base = run["accuracy"]
        acc_adj = run.get("accuracy_adjusted", acc_base)
        delta = acc_adj - acc_base
        parse_fail = run["parse_fail_rate"]
        
        # Estimate recovered cases (simplified)
        recovered = int(delta * n)
        
        md_lines.append(
            f"| {model} | {setting} | {split} | {n} | {acc_base:.6f} | {acc_adj:.6f} | {delta:.6f} | {parse_fail:.6f} | {recovered} |"
        )
        
        if acc_adj > best_acc:
            best_acc = acc_adj
            best_run = run
        if acc_adj < worst_acc:
            worst_acc = acc_adj
            worst_run = run
    
    md_lines.extend([
        "",
        "## Key Findings",
    ])
    
    if best_run and worst_run:
        md_lines.extend([
            f"- Best adjusted run: {best_run['model']} / {best_run['setting']} @ {best_acc:.4f}",
            f"- Worst adjusted run: {worst_run['model']} / {worst_run['setting']} @ {worst_acc:.4f}",
        ])
    
    md_lines.extend([
        "- Percent auto-scale recovers a non-trivial number of errors, especially for 4B.",
        "- Parse failures are now low; dominant failures are numeric reasoning and scale/selection mistakes.",
        "",
    ])
    
    if error_breakdown:
        md_lines.extend([
            "## Error Breakdown (Aggregated Latest Runs)",
        ])
        total_errors = sum(error_breakdown.values())
        for error_type, count in sorted(error_breakdown.items(), key=lambda x: -x[1]):
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            md_lines.append(f"- {error_type}: {count} ({percentage:.1f}%)")
        md_lines.append("")
    
    md_lines.extend([
        "## Interpretation",
        "- Disabling thinking mode removed most parse-fail problems from long thought traces.",
        "- Remaining gap is primarily answer correctness, not answer extractability.",
        "- Full-context setting can still hurt due to distractor context and wrong number selection.",
        "",
        "## Next-Step Recommendations",
        "1. Keep adjusted and base metrics both reported for transparency.",
        "2. Add stricter output format constraints (single numeric token regex guard).",
        "3. Introduce lightweight verifier/re-ranking after baseline reporting is finalized.",
        "4. Evaluate per-question-type slices (percent, ratio, difference, multi-step arithmetic).",
    ])
    
    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"Report generated: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate final report from evaluation results")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--output", type=str, default="results/final_report.md", help="Output report path")
    
    args = parser.parse_args()
    
    summary = load_summary(args.results_dir)
    error_breakdown = load_error_breakdown(args.results_dir)
    generate_report(summary, error_breakdown, args.output)


if __name__ == "__main__":
    main()
