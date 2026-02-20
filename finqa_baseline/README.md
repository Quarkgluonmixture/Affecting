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

## Cache (important on RunPod)

```bash
mkdir -p /workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
```

## FinQA eval

Thinking is disabled by default (`--enable_thinking` not set).

```bash
python eval_finqa.py \
  --model_name Qwen/Qwen3-8B \
  --split test \
  --setting oracle \
  --cache_dir /workspace/.cache/huggingface \
  --max_new_tokens 32
```

```bash
python eval_finqa.py \
  --model_name Qwen/Qwen3-8B \
  --split test \
  --setting full \
  --cache_dir /workspace/.cache/huggingface \
  --max_new_tokens 32
```

## New metric option: percent auto-scale

Enabled by default (`--percent_auto_scale`).

Rule: if question is percentage-like and `|gold| < 1 < |pred|`, evaluator also tests `pred/100`.

- `accuracy`: adjusted accuracy (with percent auto-scale if enabled)
- `accuracy_base`: strict original accuracy
- `parse_fail_rate`: unchanged

Disable with:

```bash
python eval_finqa.py ... --no-percent_auto_scale
```

## Outputs

- `results/summary.json`
- `results/finqa_<model>_<setting>_<split>.jsonl`
- `results/error_cases.md`
