#!/usr/bin/env python3
"""
Prefetch Hugging Face model repos into local cache before training starts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefetch Hugging Face models to local cache.")
    p.add_argument("--models", type=str, nargs="+", required=True, help="Model repo ids")
    p.add_argument("--cache_dir", type=str, required=True, help="HF cache root")
    p.add_argument("--token", type=str, default="", help="HF token; optional")
    p.add_argument("--allow-pattern", dest="allow_patterns", action="append", default=[])
    p.add_argument("--max-workers", type=int, default=8)
    return p.parse_args()


def _effective_token(cli_token: str) -> str:
    if cli_token.strip():
        return cli_token.strip()
    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token
    return ""


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    token = _effective_token(args.token)
    model_results: List[Dict[str, Any]] = []

    for model in args.models:
        print(f"[prefetch] downloading {model} ...")
        local_path = snapshot_download(
            repo_id=model,
            cache_dir=str(cache_dir),
            token=token or None,
            allow_patterns=(args.allow_patterns if args.allow_patterns else None),
            max_workers=int(args.max_workers),
            resume_download=True,
        )
        model_results.append({"model": model, "local_path": local_path})
        print(f"[prefetch] ready: {model} -> {local_path}")

    payload = {
        "cache_dir": str(cache_dir),
        "token_used": bool(token),
        "results": model_results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

