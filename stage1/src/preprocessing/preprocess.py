"""
preprocess.py — Convert raw data dicts into training-ready {"text": ...} format.

This bridges Member 4's prompting logic into the main pipeline.
Member 4 should update the prompt template as needed.
"""

import sys
import os

# Add project root to path so we can import scripts/prompting.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from prompting import build_prompt


def preprocess_data(data, config):
    """
    Convert raw data list into [{"text": "..."}] for SFTTrainer.

    Args:
        data: list of dicts with keys: id, question, context, answer, source_dataset
        config: full config dict

    Returns:
        list of {"text": "full prompt string"} dicts
    """
    preprocessing_cfg = config.get("preprocessing", {})
    thinking = bool(preprocessing_cfg.get("thinking", False))
    supervision_style = str(preprocessing_cfg.get("supervision_style", "answer_only"))
    final_answer_tag = str(preprocessing_cfg.get("final_answer_tag", "FINAL_ANSWER"))

    processed = []
    for example in data:
        text = build_prompt(
            example,
            thinking=thinking,
            supervision_style=supervision_style,
            final_answer_tag=final_answer_tag,
        )
        processed.append({"text": text})

    return processed
