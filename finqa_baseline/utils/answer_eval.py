from __future__ import annotations

from typing import Any, Dict

import regex as re


def extract_final_answer_text(
    raw_output: str,
    answer_format: str = "final_answer_tag",
    final_answer_tag: str = "FINAL_ANSWER",
) -> Dict[str, str]:
    text = (raw_output or "").strip()
    if answer_format != "final_answer_tag":
        return {"final_answer_text": text, "tag_status": "absent"}

    open_tag = f"[{final_answer_tag}]"
    close_tag = f"[/{final_answer_tag}]"

    pattern = re.compile(re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), flags=re.S)
    matches = pattern.findall(text)
    if matches:
        return {"final_answer_text": str(matches[-1]).strip(), "tag_status": "closed"}

    idx = text.rfind(open_tag)
    if idx >= 0:
        tail = text[idx + len(open_tag) :].strip()
        return {"final_answer_text": tail, "tag_status": "open_only"}

    return {"final_answer_text": text, "tag_status": "absent"}


def _import_math_verify():
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "math-verify is required for evaluator=math_verify. "
            "Install with: pip install math-verify==0.9.0"
        ) from e

    return ExprExtractionConfig, LatexExtractionConfig, parse, verify


def evaluate_mathverify(
    gold_text: str,
    pred_text: str,
    float_rounding: int = 6,
) -> Dict[str, Any]:
    if not str(gold_text or "").strip():
        return {"correct": False, "parse_fail": True, "error": "empty_gold_text"}

    if not str(pred_text or "").strip():
        return {"correct": False, "parse_fail": True, "error": "empty_pred_text"}

    ExprExtractionConfig, LatexExtractionConfig, parse, verify = _import_math_verify()

    extraction_cfg = [LatexExtractionConfig(), ExprExtractionConfig()]

    try:
        pred_candidates = parse(
            str(pred_text),
            extraction_config=extraction_cfg,
            fallback_mode="first_match",
            extraction_mode="any_match",
            parsing_timeout=5,
            raise_on_error=False,
        )
        gold_candidates = parse(
            str(gold_text),
            extraction_config=extraction_cfg,
            fallback_mode="first_match",
            extraction_mode="any_match",
            parsing_timeout=5,
            raise_on_error=False,
        )
    except Exception as e:
        return {"correct": False, "parse_fail": True, "error": f"parse_error:{e}"}

    if not pred_candidates:
        return {"correct": False, "parse_fail": True, "error": "pred_parse_empty"}
    if not gold_candidates:
        return {"correct": False, "parse_fail": True, "error": "gold_parse_empty"}

    try:
        correct = bool(
            verify(
                gold_candidates,
                pred_candidates,
                float_rounding=float_rounding,
                timeout_seconds=5,
                raise_on_error=False,
            )
        )
        return {"correct": correct, "parse_fail": False, "error": ""}
    except Exception as e:
        return {"correct": False, "parse_fail": True, "error": f"verify_error:{e}"}


def ensure_mathverify_installed() -> None:
    _import_math_verify()
