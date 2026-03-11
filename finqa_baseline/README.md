# finqa_baseline

Reproducible zero-shot baseline evaluation pipeline for COMP0087.

## Setup

```bash
cd /workspace/finqa_baseline
bash setup.sh
source .venv/bin/activate
```

If FinQA loading fails due datasets script policy:

```bash
pip install -U "datasets<4"
```

## Cache (important on RunPod/SageMaker)

```bash
mkdir -p /workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
```

## New baseline defaults (FinQA)

`eval_finqa.py` now defaults to:

- `thinking=true`
- answer format: `[FINAL_ANSWER]...[/FINAL_ANSWER]`
- primary evaluator: `math_verify`
- `max_new_tokens=256`

Legacy numeric evaluator is still computed and reported for side-by-side comparison.

## FinQA eval examples

```bash
python eval_finqa.py \
  --model_name Qwen/Qwen3-8B \
  --split test \
  --setting oracle \
  --cache_dir /workspace/.cache/huggingface
```

```bash
python eval_finqa.py \
  --model_name Qwen/Qwen3-8B \
  --split test \
  --setting full \
  --cache_dir /workspace/.cache/huggingface
```

Override evaluator/format if needed:

```bash
python eval_finqa.py ... --evaluator numeric_legacy --answer_format plain_numeric
```

## One-command pipeline (recommended)

Runs:

1. regression checks (`FINAL_ANSWER` extraction + Math-Verify)
2. 4 FinQA runs (`Qwen3-8B/4B x oracle/full`)
3. final markdown report

```bash
bash run_all_evals.sh
```

## Outputs

- `results/summary.json`
- `results/finqa_<model>_<setting>_<split>.jsonl`
- `results/regression_final_answer_mathverify.md`
- `results/regression_final_answer_mathverify.json`
- `results/error_cases.md`
- `results/final_report.md`

## Key result fields

Per-run summary now includes:

- `accuracy_mathverify`, `parse_fail_rate_mathverify`
- `accuracy_legacy`, `accuracy_legacy_base`, `parse_fail_rate_legacy`
- `tag_status_counts` (`closed` / `open_only` / `absent`)
- `enable_thinking`, `answer_format`, `final_answer_tag`, `evaluator`
