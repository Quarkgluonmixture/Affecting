#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List

from utils import ensure_dir, evaluate_mathverify, extract_final_answer_text, extract_numeric_prediction


def _run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    extraction = extract_final_answer_text(
        raw_output=case["raw_output"],
        answer_format="final_answer_tag",
        final_answer_tag="FINAL_ANSWER",
    )
    final_answer_text = extraction["final_answer_text"]
    tag_status = extraction["tag_status"]

    mv_eval = evaluate_mathverify(gold_text=case["gold_text"], pred_text=final_answer_text)
    legacy_pred = extract_numeric_prediction(final_answer_text)
    legacy_ok = (legacy_pred is not None) and abs(float(legacy_pred) - float(case["gold_numeric"])) < 1e-6

    passed = (
        tag_status == case["expect_tag_status"]
        and bool(mv_eval["correct"]) == bool(case["expect_mathverify"])
        and bool(legacy_ok) == bool(case["expect_legacy"])
    )

    return {
        "name": case["name"],
        "passed": passed,
        "tag_status": tag_status,
        "final_answer_text": final_answer_text,
        "mathverify_correct": bool(mv_eval["correct"]),
        "mathverify_parse_fail": bool(mv_eval["parse_fail"]),
        "legacy_pred": legacy_pred,
        "legacy_correct": bool(legacy_ok),
        "error": str(mv_eval.get("error", "")),
    }


def _build_cases() -> List[Dict[str, Any]]:
    return [
        {
            "name": "multi-number noise, tagged answer wins",
            "raw_output": "Intermediates: 12, 24, 99. [FINAL_ANSWER]37.1[/FINAL_ANSWER]",
            "gold_text": "37.1",
            "gold_numeric": 37.1,
            "expect_tag_status": "closed",
            "expect_mathverify": True,
            "expect_legacy": True,
        },
        {
            "name": "truncated output with open tag only",
            "raw_output": "Reasoning... [FINAL_ANSWER]0.42",
            "gold_text": "0.42",
            "gold_numeric": 0.42,
            "expect_tag_status": "open_only",
            "expect_mathverify": True,
            "expect_legacy": True,
        },
        {
            "name": "percent vs decimal equivalence",
            "raw_output": "[FINAL_ANSWER]42%[/FINAL_ANSWER]",
            "gold_text": "0.42",
            "gold_numeric": 0.42,
            "expect_tag_status": "closed",
            "expect_mathverify": True,
            "expect_legacy": True,
        },
        {
            "name": "no tag fallback does not crash",
            "raw_output": "Candidates: 1, 2, and finally 3.5",
            "gold_text": "3.5",
            "gold_numeric": 3.5,
            "expect_tag_status": "absent",
            "expect_mathverify": True,
            "expect_legacy": True,
        },
    ]


def _write_markdown(path: str, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Regression Check: FINAL_ANSWER + Math-Verify",
        "",
        "| case | passed | tag_status | mathverify_correct | mathverify_parse_fail | legacy_pred | legacy_correct |",
        "|---|---|---|---|---|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['passed']} | {r['tag_status']} | {r['mathverify_correct']} | "
            f"{r['mathverify_parse_fail']} | {r['legacy_pred']} | {r['legacy_correct']} |"
        )
    lines.append("")
    lines.append(f"- pass_count: {sum(1 for r in rows if r['passed'])}/{len(rows)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression checks for FINAL_ANSWER extraction and math-verify.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--output",
        type=str,
        default="results/regression_final_answer_mathverify.md",
        help="Markdown summary output path",
    )
    args = parser.parse_args()

    ensure_dir(args.results_dir)

    cases = _build_cases()
    rows = [_run_case(c) for c in cases]

    _write_markdown(args.output, rows)

    summary_path = os.path.join(args.results_dir, "regression_final_answer_mathverify.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": len(rows),
                "passed": sum(1 for r in rows if r["passed"]),
                "failed": [r for r in rows if not r["passed"]],
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    failed = [r for r in rows if not r["passed"]]
    if failed:
        print(json.dumps({"status": "failed", "failed_cases": failed}, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    print(json.dumps({"status": "ok", "passed": len(rows), "total": len(rows)}, ensure_ascii=False, indent=2))
    print(f"[saved] {args.output}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
