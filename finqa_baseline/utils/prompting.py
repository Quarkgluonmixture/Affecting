from typing import Iterable, List, Sequence


def build_system_instruction() -> str:
    return (
        "You are a financial QA assistant. "
        "ONLY output the final numeric answer; no explanation; no units."
    )


def _safe_join_lines(lines: Iterable[str], sep: str = "\n") -> str:
    cleaned: List[str] = []
    for line in lines:
        if line is None:
            continue
        s = str(line).strip()
        if s:
            cleaned.append(s)
    return sep.join(cleaned)


def _table_to_text(table) -> str:
    if table is None:
        return ""

    if isinstance(table, str):
        return table

    if isinstance(table, dict):
        headers = table.get("header") or table.get("headers") or []
        rows = table.get("rows") or table.get("table") or []
        parts = []
        if headers:
            parts.append(" | ".join(str(h) for h in headers))
        for r in rows:
            if isinstance(r, (list, tuple)):
                parts.append(" | ".join(str(c) for c in r))
            else:
                parts.append(str(r))
        return "\n".join(parts)

    if isinstance(table, Sequence):
        parts = []
        for r in table:
            if isinstance(r, (list, tuple)):
                parts.append(" | ".join(str(c) for c in r))
            else:
                parts.append(str(r))
        return "\n".join(parts)

    return str(table)


def _pick_gold_evidence(example: dict) -> List[str]:
    candidates = []

    gold_inds = example.get("gold_inds")
    pre = example.get("pre_text") or []
    post = example.get("post_text") or []

    if isinstance(gold_inds, dict):
        for _, text in gold_inds.items():
            candidates.append(str(text))
    elif isinstance(gold_inds, list):
        for item in gold_inds:
            if isinstance(item, int):
                if 0 <= item < len(pre):
                    candidates.append(str(pre[item]))
                elif 0 <= (item - len(pre)) < len(post):
                    candidates.append(str(post[item - len(pre)]))
            elif isinstance(item, str):
                if item.isdigit():
                    idx = int(item)
                    if 0 <= idx < len(pre):
                        candidates.append(str(pre[idx]))
                    elif 0 <= (idx - len(pre)) < len(post):
                        candidates.append(str(post[idx - len(pre)]))
                    else:
                        candidates.append(item)
                else:
                    candidates.append(item)
    elif isinstance(gold_inds, str):
        candidates.append(gold_inds)

    if not candidates:
        for k in ["evidence", "evidences", "gold_evidence", "gold_text"]:
            val = example.get(k)
            if isinstance(val, list):
                candidates.extend(str(v) for v in val)
            elif isinstance(val, str):
                candidates.append(val)

    return [c.strip() for c in candidates if str(c).strip()]


def build_finqa_prompt(example: dict, setting: str) -> str:
    question = str(example.get("question", "")).strip()
    pre_text = _safe_join_lines(example.get("pre_text") or [])
    post_text = _safe_join_lines(example.get("post_text") or [])
    table_text = _table_to_text(example.get("table"))

    if setting == "oracle":
        evidence = _safe_join_lines(_pick_gold_evidence(example))
        context = evidence if evidence else "[NO_ORACLE_EVIDENCE_FOUND]"
    else:
        context = _safe_join_lines(
            [
                "[PRE_TEXT]",
                pre_text,
                "[TABLE]",
                table_text,
                "[POST_TEXT]",
                post_text,
            ]
        )

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Return ONLY the final numeric answer; no explanation; no units."
    )
    return prompt


def build_carbonpdf_prompt(example: dict) -> str:
    question = str(example.get("question", "")).strip()
    context = str(example.get("context", "")).strip()
    return (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Return ONLY the final numeric answer; no explanation; no units."
    )
