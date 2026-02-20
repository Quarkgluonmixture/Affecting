import math
from typing import Any, Optional

import regex as re

_NUM_PATTERN = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|[-+]?\d+(?:\.\d+)?%?")


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)
    s = str(v).strip()
    if not s:
        return None

    s = s.replace("$", "").replace(",", "")
    if s.endswith("%"):
        core = s[:-1].strip()
        try:
            return float(core) / 100.0
        except Exception:
            return None

    try:
        return float(s)
    except Exception:
        return None


def normalize_gold_numeric(example: dict) -> Optional[float]:
    for key in ["answer", "ans", "gold", "label", "target", "exe_ans"]:
        if key in example:
            val = _to_float(example.get(key))
            if val is not None:
                return val

    qa = example.get("qa")
    if isinstance(qa, dict):
        for key in ["answer", "exe_ans", "ans"]:
            val = _to_float(qa.get(key))
            if val is not None:
                return val

    return None


def extract_numeric_prediction(text: str) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    matches = _NUM_PATTERN.findall(s)
    if not matches:
        return None

    last = matches[-1].replace(",", "").strip()
    if last.endswith("%"):
        core = last[:-1].strip()
        try:
            return float(core) / 100.0
        except Exception:
            return None

    try:
        return float(last)
    except Exception:
        return None


def is_correct_numeric(pred: Optional[float], gold: Optional[float], atol: float, rtol: float) -> bool:
    if pred is None or gold is None:
        return False
    return math.isclose(pred, gold, abs_tol=atol, rel_tol=rtol)
