# finqa_baseline

Reproducible zero-shot baseline evaluation pipeline for COMP0087.

## Setup

```bash
cd /home/jiaming/workspace/FinQA/finqa_baseline
bash setup.sh
source .venv/bin/activate
```

## Cache (important on shared machines)

```bash
mkdir -p /home/jiaming/workspace/.cache/huggingface/{hub,transformers}
export HF_HOME=/home/jiaming/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
```

## DGX Spark quirks (authoritative)

All runs on this machine must follow:
- `/home/jiaming/workspace/FinQA/DGX_SPARK_MACHINE_QUIRKS.md`

The run scripts now auto-apply the required env via:
- `scripts/apply_dgx_spark_quirks.sh`

Equivalent variables:

```bash
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export HF_HUB_DISABLE_XET=1
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
  --cache_dir /home/jiaming/workspace/.cache/huggingface
```

```bash
python eval_finqa.py \
  --model_name Qwen/Qwen3-8B \
  --split test \
  --setting full \
  --cache_dir /home/jiaming/workspace/.cache/huggingface
```

Override evaluator/format if needed:

```bash
python eval_finqa.py ... --evaluator numeric_legacy --answer_format plain_numeric
```

Evaluate a LoRA adapter on top of a base model:

```bash
python eval_finqa.py \
  --model_name Qwen/Qwen3-8B \
  --adapter_path /path/to/checkpoint-last \
  --setting oracle \
  --no-enable_thinking \
  --answer_format final_answer_tag \
  --final_answer_tag FINAL_ANSWER
```

## Robust verification matrix (thinking true + false)

Runs:

1. regression checks
2. 8 FinQA runs (`Qwen3-8B/4B x oracle/full x thinking true/false`)
3. per-mode reports + robust verification summary

```bash
bash run_verification_matrix.sh
```

Main outputs:

- `results/robust_verification_report.md`
- `results/robust_verification_summary.json`

## GPU guard auto-start (optional)

```bash
nohup bash gpu_guard_autorun.sh > logs/gpu_guard_console.log 2>&1 &
echo $! > logs/gpu_guard_autorun.pid
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
