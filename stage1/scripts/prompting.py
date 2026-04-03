from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

CURRENT_DIR = Path(__file__).resolve().parent
STAGE1_ROOT = CURRENT_DIR.parent
if str(STAGE1_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE1_ROOT))

from src.preprocessing.formula_utils import derive_formula_metadata


def _pick_target_answer(example: Dict[str, Any]) -> str:
    for key in ["target_answer", "corrected_answer", "answer", "output", "final_answer"]:
        if key in example and example.get(key) is not None:
            text = str(example.get(key)).strip()
            if text:
                return text
    return ""


def _pick_formula_expression(example: Dict[str, Any]) -> str:
    for key in ["formula_expression", "program_expression"]:
        if key in example and example.get(key) is not None:
            text = str(example.get(key)).strip()
            if text:
                return text

    # Fallback for raw unified jsonl without pre-built formula fields.
    meta = derive_formula_metadata(example)
    text = str(meta.get("formula_expression", "")).strip()
    if text:
        return text

    raw_program = str(example.get("program", "")).strip()
    if raw_program:
        return raw_program
    return "N/A"


def _render_final_answer(answer: str, final_answer_tag: str) -> str:
    return f"[{final_answer_tag}]{answer}[/{final_answer_tag}]"


def build_prompt(
    example: Dict[str, Any],
    thinking: bool = False,
    supervision_style: str = "answer_only",
    final_answer_tag: str = "FINAL_ANSWER",
) -> str:
    del thinking  # no-thinking protocol is locked; keep arg for compatibility.

    question = str(example.get("question", ""))
    context = str(example.get("context", ""))
    answer = _pick_target_answer(example)

    instruction = (
        "Solve the financial numerical reasoning problem. "
        f"Return exactly one tagged final answer as [{final_answer_tag}]...[/"
        f"{final_answer_tag}] in the output."
    )

    input_text = f"""Context:
{context}

Question:
{question}
"""

    if supervision_style == "formula_rationale":
        formula_expr = _pick_formula_expression(example)
        output_text = (
            f"Formula: {formula_expr}\n"
            f"{_render_final_answer(answer, final_answer_tag)}\n"
        )
    else:
        output_text = f"{_render_final_answer(answer, final_answer_tag)}\n"

    return f"""Instruction:
{instruction}

Input:
{input_text}

Output:
{output_text}
"""
