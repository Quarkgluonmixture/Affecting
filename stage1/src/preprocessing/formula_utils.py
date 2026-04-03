"""
Utilities for converting FinQA-style programs into executable formulas.

This module is used by:
- data target builder (offline audit / correction)
- prompting (online fallback formula rendering)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


_FUNC_RE = re.compile(r"^([A-Za-z_]+)\((.*)\)$")
_REF_RE = re.compile(r"^#(\d+)$")
_NUM_RE = re.compile(r"^[-+]?\d*\.?\d+%?$")


def split_top_level_steps(program: str) -> List[str]:
    text = (program or "").strip()
    if not text:
        return []

    steps: List[str] = []
    depth = 0
    cur: List[str] = []

    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1

        if ch == "," and depth == 0:
            token = "".join(cur).strip()
            if token:
                steps.append(token)
            cur = []
        else:
            cur.append(ch)

    tail = "".join(cur).strip()
    if tail:
        steps.append(tail)
    return steps


def _split_step_args(args_raw: str) -> List[str]:
    if not args_raw:
        return []
    parts = [part.strip() for part in args_raw.split(",")]
    return [p for p in parts if p]


def _parse_const(token: str) -> Optional[float]:
    if not token.startswith("const_"):
        return None
    payload = token[len("const_") :].strip()
    if not payload:
        return None
    if payload.startswith("m") and payload[1:].isdigit():
        return -float(payload[1:])
    try:
        return float(payload)
    except Exception:
        return None


def parse_number(token: str) -> Optional[float]:
    if token is None:
        return None
    s = str(token).strip()
    if not s:
        return None

    if s.startswith("(") and s.endswith(")") and len(s) > 2:
        core = s[1:-1].strip()
        if _NUM_RE.match(core):
            s = "-" + core

    s = s.replace("$", "").replace(",", "")

    const_val = _parse_const(s)
    if const_val is not None:
        return const_val

    if s.endswith("%"):
        body = s[:-1].strip()
        try:
            return float(body) / 100.0
        except Exception:
            return None

    try:
        return float(s)
    except Exception:
        return None


def format_number(value: float) -> str:
    if value is None:
        return ""
    if math.isnan(value) or math.isinf(value):
        return ""
    s = f"{float(value):.12g}"
    if "e" in s or "E" in s:
        return s
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _is_close(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)


def classify_scale_relation(gold_value: Optional[float], formula_value: Optional[float]) -> str:
    if gold_value is None or formula_value is None:
        return "mismatch"
    if _is_close(gold_value, formula_value):
        return "consistent"
    if _is_close(gold_value * 100.0, formula_value):
        return "x100_match"
    if not _is_close(gold_value, 0.0) and _is_close(gold_value / 100.0, formula_value):
        return "div100_match"
    return "mismatch"


@dataclass
class StepState:
    expression: str
    value: Optional[float]


def _resolve_arg(token: str, states: List[StepState]) -> StepState:
    raw = token.strip()
    if not raw:
        return StepState(expression="", value=None)

    ref_match = _REF_RE.match(raw)
    if ref_match:
        ref_idx = int(ref_match.group(1))
        if 0 <= ref_idx < len(states):
            return states[ref_idx]
        return StepState(expression=raw, value=None)

    parsed = parse_number(raw)
    if parsed is not None:
        if raw.startswith("const_"):
            return StepState(expression=format_number(parsed), value=parsed)
        return StepState(expression=raw, value=parsed)
    return StepState(expression=raw, value=None)


def _render_expr(op: str, args: List[str]) -> str:
    if not args:
        return ""
    if op == "add":
        return f"({' + '.join(args)})"
    if op == "subtract":
        if len(args) == 1:
            return args[0]
        return f"({args[0]} - {args[1]})"
    if op == "multiply":
        return f"({' * '.join(args)})"
    if op == "divide":
        if len(args) == 1:
            return args[0]
        return f"({args[0]} / {args[1]})"
    if op == "exp":
        if len(args) == 1:
            return args[0]
        return f"({args[0]} ^ {args[1]})"
    if op == "greater":
        if len(args) == 1:
            return args[0]
        return f"max({args[0]}, {args[1]})"
    if op == "table_sum":
        return f"({' + '.join(args)})" if len(args) > 1 else args[0]
    if op == "table_average":
        if len(args) == 1:
            return args[0]
        return f"(({' + '.join(args)}) / {len(args)})"
    if op == "table_max":
        return f"max({', '.join(args)})"
    if op == "table_min":
        return f"min({', '.join(args)})"
    return f"{op}({', '.join(args)})"


def _compute_value(op: str, vals: List[Optional[float]]) -> Optional[float]:
    if not vals or any(v is None for v in vals):
        return None

    nums = [float(v) for v in vals if v is not None]
    if op == "add":
        return sum(nums)
    if op == "subtract":
        if len(nums) == 1:
            return nums[0]
        return nums[0] - nums[1]
    if op == "multiply":
        out = 1.0
        for v in nums:
            out *= v
        return out
    if op == "divide":
        if len(nums) == 1:
            return nums[0]
        if _is_close(nums[1], 0.0):
            return None
        return nums[0] / nums[1]
    if op == "exp":
        if len(nums) == 1:
            return nums[0]
        try:
            return float(nums[0] ** nums[1])
        except Exception:
            return None
    if op == "greater":
        if len(nums) == 1:
            return nums[0]
        return nums[0] if nums[0] >= nums[1] else nums[1]
    if op == "table_sum":
        return sum(nums)
    if op == "table_average":
        if not nums:
            return None
        return sum(nums) / len(nums)
    if op == "table_max":
        return max(nums) if nums else None
    if op == "table_min":
        return min(nums) if nums else None
    return None


def execute_program(program: str) -> Dict[str, object]:
    steps = split_top_level_steps(program)
    if not steps:
        return {
            "formula_expression": "",
            "formula_value": None,
            "program_steps": 0,
            "formula_exec_ok": False,
            "formula_error": "empty_program",
        }

    states: List[StepState] = []
    op_steps = 0

    for step in steps:
        m = _FUNC_RE.match(step)
        if not m:
            literal_value = parse_number(step)
            states.append(StepState(expression=step, value=literal_value))
            continue

        op = m.group(1).lower()
        op_steps += 1
        arg_tokens = _split_step_args(m.group(2))
        arg_states = [_resolve_arg(tok, states) for tok in arg_tokens]
        arg_exprs = [st.expression for st in arg_states]
        arg_vals = [st.value for st in arg_states]

        expr = _render_expr(op, arg_exprs)
        val = _compute_value(op, arg_vals)
        states.append(StepState(expression=expr if expr else step, value=val))

    final_state = states[-1]
    steps_out = op_steps if op_steps > 0 else 1
    return {
        "formula_expression": final_state.expression.strip(),
        "formula_value": final_state.value,
        "program_steps": steps_out,
        "formula_exec_ok": final_state.value is not None,
        "formula_error": "" if final_state.value is not None else "non_numeric_or_unsupported_program",
    }


def program_steps_bucket(program_steps: int) -> str:
    if program_steps <= 1:
        return "single"
    if program_steps == 2:
        return "double"
    return "multi"


def derive_formula_metadata(sample: Dict[str, object]) -> Dict[str, object]:
    gold_raw = str(sample.get("answer", "") if sample.get("answer") is not None else "").strip()
    gold_value = parse_number(gold_raw)

    exec_result = execute_program(str(sample.get("program", "") or ""))
    formula_value = exec_result["formula_value"]
    scale_relation = classify_scale_relation(gold_value=gold_value, formula_value=formula_value)

    answer_corrected = bool(
        (gold_value is not None)
        and (formula_value is not None)
        and (scale_relation != "consistent")
    )
    if answer_corrected:
        corrected_answer = format_number(float(formula_value))
    else:
        corrected_answer = gold_raw

    return {
        "gold_raw": gold_raw,
        "gold_value": gold_value,
        "formula_expression": exec_result["formula_expression"],
        "formula_value": formula_value,
        "formula_exec_ok": bool(exec_result["formula_exec_ok"]),
        "formula_error": str(exec_result["formula_error"]),
        "program_steps": int(exec_result["program_steps"]),
        "program_steps_bucket": program_steps_bucket(int(exec_result["program_steps"])),
        "scale_relation": scale_relation,
        "answer_corrected": answer_corrected,
        "corrected_answer": corrected_answer,
    }
