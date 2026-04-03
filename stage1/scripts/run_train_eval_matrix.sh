#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_ROOT="${STAGE1_ROOT}/../finqa_baseline"

if [[ -x "${STAGE1_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${STAGE1_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "[error] no usable python found for stage1." >&2
  exit 127
fi

if [[ -x "${BASELINE_ROOT}/.venv/bin/python" ]]; then
  EVAL_PYTHON_BIN="${BASELINE_ROOT}/.venv/bin/python"
else
  EVAL_PYTHON_BIN="${PYTHON_BIN}"
fi

if [[ -f "${BASELINE_ROOT}/scripts/apply_dgx_spark_quirks.sh" ]]; then
  # shellcheck source=/dev/null
  source "${BASELINE_ROOT}/scripts/apply_dgx_spark_quirks.sh"
fi

DATA_UNIFIED_DIR="${DATA_UNIFIED_DIR:-${STAGE1_ROOT}/data/unified}"
DERIVED_DIR="${DERIVED_DIR:-${DATA_UNIFIED_DIR}/derived}"
SUBSET_DIR="${SUBSET_DIR:-${DERIVED_DIR}/subsets}"
CONFIG_DIR="${CONFIG_DIR:-${STAGE1_ROOT}/configs/generated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STAGE1_ROOT}/outputs/sft_matrix}"
RESULTS_ROOT="${RESULTS_ROOT:-${BASELINE_ROOT}/results/sft_matrix}"
MANIFEST_PATH="${MANIFEST_PATH:-${RESULTS_ROOT}/run_manifest.jsonl}"

TRAIN_RAW="${TRAIN_RAW:-${DATA_UNIFIED_DIR}/train.jsonl}"
DEV_RAW="${DEV_RAW:-${DATA_UNIFIED_DIR}/dev.jsonl}"
DEBUG_RAW="${DEBUG_RAW:-${DATA_UNIFIED_DIR}/debug.jsonl}"

TRAIN_NORM="${TRAIN_NORM:-${DERIVED_DIR}/train_norm.jsonl}"
DEV_NORM="${DEV_NORM:-${DERIVED_DIR}/dev_norm.jsonl}"
DEBUG_NORM="${DEBUG_NORM:-${DERIVED_DIR}/debug_norm.jsonl}"

SEED="${SEED:-42}"
EVAL_SETTING="${EVAL_SETTING:-oracle}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:--1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"

MODEL_4B="${MODEL_4B:-Qwen/Qwen3-4B}"
MODEL_8B="${MODEL_8B:-Qwen/Qwen3-8B}"
USE_QLORA_4B="${USE_QLORA_4B:-false}"
USE_QLORA_8B="${USE_QLORA_8B:-true}"

DEBUG_MAX_STEPS="${DEBUG_MAX_STEPS:-8}"
FULL_MAX_STEPS="${FULL_MAX_STEPS:--1}"
RUN_4B_ABLATION="${RUN_4B_ABLATION:-false}"

mkdir -p "${DERIVED_DIR}" "${SUBSET_DIR}" "${CONFIG_DIR}" "${OUTPUT_ROOT}" "${RESULTS_ROOT}"
rm -f "${MANIFEST_PATH}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

sanitize_model() {
  local model="$1"
  echo "${model}" | tr '/:' '__'
}

build_targets() {
  log "Building audited targets with percent-scale consistency checks..."
  "${PYTHON_BIN}" "${STAGE1_ROOT}/scripts/build_formula_rationale_targets.py" \
    --input_jsonl "${TRAIN_RAW}" \
    --output_jsonl "${TRAIN_NORM}" \
    --summary_json "${DERIVED_DIR}/train_norm.summary.json"
  "${PYTHON_BIN}" "${STAGE1_ROOT}/scripts/build_formula_rationale_targets.py" \
    --input_jsonl "${DEV_RAW}" \
    --output_jsonl "${DEV_NORM}" \
    --summary_json "${DERIVED_DIR}/dev_norm.summary.json"
  "${PYTHON_BIN}" "${STAGE1_ROOT}/scripts/build_formula_rationale_targets.py" \
    --input_jsonl "${DEBUG_RAW}" \
    --output_jsonl "${DEBUG_NORM}" \
    --summary_json "${DERIVED_DIR}/debug_norm.summary.json"
}

build_subsets() {
  log "Building nested stratified subsets (250 ⊂ 1000 ⊂ full)..."
  "${PYTHON_BIN}" "${STAGE1_ROOT}/scripts/build_stratified_subsets.py" \
    --input_jsonl "${TRAIN_NORM}" \
    --output_dir "${SUBSET_DIR}" \
    --sizes 250 1000 \
    --seed "${SEED}" \
    --prefix "train"
}

write_config() {
  local run_name="$1"
  local model_name="$2"
  local supervision_style="$3"
  local train_file="$4"
  local mode="$5"          # debug | full
  local use_qlora="$6"
  local out_cfg="${CONFIG_DIR}/${run_name}.yaml"

  "${PYTHON_BIN}" - "${STAGE1_ROOT}" "${run_name}" "${model_name}" "${supervision_style}" "${train_file}" "${mode}" "${use_qlora}" "${DEV_NORM}" "${OUTPUT_ROOT}" "${DEBUG_MAX_STEPS}" "${FULL_MAX_STEPS}" "${SEED}" "${out_cfg}" <<'PY'
import sys
from pathlib import Path
import yaml

(
    stage1_root,
    run_name,
    model_name,
    supervision_style,
    train_file,
    mode,
    use_qlora,
    dev_norm,
    output_root,
    debug_max_steps,
    full_max_steps,
    seed,
    out_cfg,
) = sys.argv[1:]

stage1_root = Path(stage1_root)
base_cfg = stage1_root / "configs" / ("debug.yaml" if mode == "debug" else "full.yaml")
cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))

output_dir = Path(output_root) / run_name
cfg["run_name"] = run_name
cfg["seed"] = int(seed)

cfg.setdefault("data", {})
cfg["data"]["train_file"] = str(train_file)
cfg["data"]["dev_file"] = str(dev_norm)
cfg["data"]["max_train_samples"] = None

cfg.setdefault("model", {})
cfg["model"]["model_name_or_path"] = model_name
cfg["model"]["use_qlora"] = str(use_qlora).lower() == "true"

cfg.setdefault("preprocessing", {})
cfg["preprocessing"]["thinking"] = False
cfg["preprocessing"]["supervision_style"] = supervision_style
cfg["preprocessing"]["final_answer_tag"] = "FINAL_ANSWER"

cfg.setdefault("training", {})
cfg["training"]["output_dir"] = str(output_dir)
if mode == "debug":
    cfg["training"]["max_steps"] = int(debug_max_steps)
    cfg["training"]["num_train_epochs"] = 1
    cfg["training"]["per_device_train_batch_size"] = 1
    cfg["training"]["gradient_accumulation_steps"] = 1
    cfg["training"]["save_steps"] = max(1, int(debug_max_steps))
    cfg["training"]["logging_steps"] = 1
else:
    if int(full_max_steps) > 0:
        cfg["training"]["max_steps"] = int(full_max_steps)

cfg.setdefault("inference", {})
cfg["inference"]["enabled"] = True
cfg["inference"]["checkpoint_dir"] = str(output_dir / "checkpoint-last")
cfg["inference"]["input_file"] = str(dev_norm)
cfg["inference"]["output_file"] = str(output_dir / "infer_predictions.jsonl")
cfg["inference"]["summary_file"] = str(output_dir / "infer_summary.json")
cfg["inference"]["max_samples"] = 3 if mode == "debug" else 10
cfg["inference"]["mode"] = "smoke_echo_gold"

cfg.setdefault("logging", {})
cfg["logging"]["log_dir"] = str(output_dir / "logs")
cfg["logging"]["train_log_file"] = str(output_dir / "logs" / "train.log")
cfg["logging"]["infer_log_file"] = str(output_dir / "logs" / "infer.log")

cfg["run_infer_after_train"] = False

Path(out_cfg).parent.mkdir(parents=True, exist_ok=True)
with open(out_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
print(out_cfg)
PY
}

append_manifest() {
  local run_name="$1"
  local phase="$2"
  local model_name="$3"
  local model_size="$4"
  local supervision_style="$5"
  local train_size="$6"
  local adapter_path="$7"
  local eval_jsonl="$8"
  local results_dir="$9"

  "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${run_name}" "${phase}" "${model_name}" "${model_size}" "${supervision_style}" "${train_size}" "${adapter_path}" "${eval_jsonl}" "${results_dir}" <<'PY'
import json
import sys
from pathlib import Path

(manifest, run_name, phase, model_name, model_size, supervision_style, train_size, adapter_path, eval_jsonl, results_dir) = sys.argv[1:]
payload = {
    "run_name": run_name,
    "phase": phase,
    "model_name": model_name,
    "model_size": model_size,
    "supervision_style": supervision_style,
    "train_size": train_size,
    "adapter_path": adapter_path,
    "eval_jsonl": eval_jsonl,
    "results_dir": results_dir,
}
path = Path(manifest)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
}

run_eval() {
  local run_name="$1"
  local phase="$2"
  local model_name="$3"
  local model_size="$4"
  local supervision_style="$5"
  local train_size="$6"
  local adapter_path="$7"
  local local_json="${8:-}"
  local num_samples_override="${9:-}"
  local append_manifest_flag="${10:-true}"

  local results_dir="${RESULTS_ROOT}/${run_name}"
  mkdir -p "${results_dir}"

  local model_safe
  model_safe="$(sanitize_model "${model_name}")"
  local eval_jsonl="${results_dir}/finqa_${model_safe}_${EVAL_SETTING}_${EVAL_SPLIT}.jsonl"

  local num_samples="${EVAL_NUM_SAMPLES}"
  if [[ -n "${num_samples_override}" ]]; then
    num_samples="${num_samples_override}"
  fi

  log "Eval run=${run_name} model=${model_name} adapter=${adapter_path:-<none>} style=${supervision_style}"
  (
    cd "${BASELINE_ROOT}"
    args=(
      eval_finqa.py
      --model_name "${model_name}"
      --setting "${EVAL_SETTING}"
      --split "${EVAL_SPLIT}"
      --cache_dir "/home/jiaming/workspace/.cache/huggingface"
      --results_dir "${results_dir}"
      --evaluator "math_verify"
      --answer_format "final_answer_tag"
      --final_answer_tag "FINAL_ANSWER"
      --no-enable_thinking
      --max_new_tokens "${EVAL_MAX_NEW_TOKENS}"
      --seed "${SEED}"
      --num_samples "${num_samples}"
    )
    if [[ -n "${adapter_path}" ]]; then
      args+=(--adapter_path "${adapter_path}")
    fi
    if [[ -n "${local_json}" ]]; then
      args+=(--local_json "${local_json}")
    fi
    "${EVAL_PYTHON_BIN}" "${args[@]}"
  )

  if [[ "${append_manifest_flag}" == "true" ]]; then
    append_manifest \
      "${run_name}" "${phase}" "${model_name}" "${model_size}" "${supervision_style}" "${train_size}" \
      "${adapter_path}" "${eval_jsonl}" "${results_dir}"
  fi
}

run_tokenizer_compat_check() {
  local run_name="$1"
  local model_name="$2"
  local adapter_path="$3"
  local compat_root="${RESULTS_ROOT}/${run_name}/compat"
  mkdir -p "${compat_root}"

  run_eval "${run_name}_compat_base" "compat" "${model_name}" "compat" "compat" "debug" "" "${DEBUG_NORM}" "4" "false"
  run_eval "${run_name}_compat_adapter" "compat" "${model_name}" "compat" "compat" "debug" "${adapter_path}" "${DEBUG_NORM}" "4" "false"

  local base_jsonl="${RESULTS_ROOT}/${run_name}_compat_base/finqa_$(sanitize_model "${model_name}")_${EVAL_SETTING}_${EVAL_SPLIT}.jsonl"
  local adapter_jsonl="${RESULTS_ROOT}/${run_name}_compat_adapter/finqa_$(sanitize_model "${model_name}")_${EVAL_SETTING}_${EVAL_SPLIT}.jsonl"

  "${PYTHON_BIN}" - "${base_jsonl}" "${adapter_jsonl}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
adapter_path = Path(sys.argv[2])
for p in [base_path, adapter_path]:
    if not p.exists():
        raise FileNotFoundError(f"compat eval output missing: {p}")

def parseable_count(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("tag_status") in {"closed", "open_only"}:
                n += 1
    return n

base_ok = parseable_count(base_path)
adapter_ok = parseable_count(adapter_path)
print(f"[compat] base_parseable={base_ok}, adapter_parseable={adapter_ok}")
if base_ok <= 0 or adapter_ok <= 0:
    raise SystemExit("compatibility check failed: no parseable FINAL_ANSWER tags")
PY
}

train_and_eval() {
  local run_name="$1"
  local phase="$2"
  local model_name="$3"
  local model_size="$4"
  local supervision_style="$5"
  local train_size="$6"
  local cfg_path="$7"

  log "Train run=${run_name} cfg=${cfg_path}"
  (
    cd "${STAGE1_ROOT}"
    bash scripts/run_train.sh "${cfg_path}"
  )

  local adapter_path="${OUTPUT_ROOT}/${run_name}/checkpoint-last"
  run_eval "${run_name}" "${phase}" "${model_name}" "${model_size}" "${supervision_style}" "${train_size}" "${adapter_path}"

  if [[ "${phase}" == "debug" ]]; then
    run_tokenizer_compat_check "${run_name}" "${model_name}" "${adapter_path}"
  fi
}

pick_winner_style() {
  "${PYTHON_BIN}" - "${RESULTS_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = {
    "answer_only": root / "8b_full_answer_only" / "summary.json",
    "formula_rationale": root / "8b_full_formula_rationale" / "summary.json",
}
metrics = {}
for style, path in candidates.items():
    if not path.exists():
        continue
    obj = json.loads(path.read_text(encoding="utf-8"))
    runs = obj.get("runs", [])
    if not runs:
        continue
    last = runs[-1]
    metrics[style] = (
        float(last.get("accuracy_adjusted", last.get("accuracy", 0.0))),
        float(last.get("parse_fail_rate_mathverify", last.get("parse_fail_rate", 1.0))),
    )

if len(metrics) < 2:
    print("answer_only")
    raise SystemExit(0)

winner = max(metrics.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]
print(winner)
PY
}

run_error_shift_report() {
  log "Running mandatory error-shift analysis..."
  "${PYTHON_BIN}" "${STAGE1_ROOT}/scripts/analyze_error_shift.py" \
    --manifest_jsonl "${MANIFEST_PATH}" \
    --output_md "${RESULTS_ROOT}/error_shift_report.md" \
    --output_json "${RESULTS_ROOT}/error_shift_report.json"
}

main() {
  log "Stage 0: data target normalization + subset generation"
  build_targets
  build_subsets

  log "Stage 1: debug matrix (4 runs)"
  for model_size in 4B 8B; do
    if [[ "${model_size}" == "4B" ]]; then
      model_name="${MODEL_4B}"
      use_qlora="${USE_QLORA_4B}"
      model_tag="4b"
    else
      model_name="${MODEL_8B}"
      use_qlora="${USE_QLORA_8B}"
      model_tag="8b"
    fi

    for style in answer_only formula_rationale; do
      run_name="${model_tag}_debug_${style}"
      cfg_path="$(write_config "${run_name}" "${model_name}" "${style}" "${DEBUG_NORM}" "debug" "${use_qlora}")"
      train_and_eval "${run_name}" "debug" "${model_name}" "${model_size}" "${style}" "debug" "${cfg_path}"
    done
  done

  log "Stage 2: full-data main matrix (4 runs)"
  for model_size in 4B 8B; do
    if [[ "${model_size}" == "4B" ]]; then
      model_name="${MODEL_4B}"
      use_qlora="${USE_QLORA_4B}"
      model_tag="4b"
    else
      model_name="${MODEL_8B}"
      use_qlora="${USE_QLORA_8B}"
      model_tag="8b"
    fi

    for style in answer_only formula_rationale; do
      run_name="${model_tag}_full_${style}"
      cfg_path="$(write_config "${run_name}" "${model_name}" "${style}" "${TRAIN_NORM}" "full" "${use_qlora}")"
      train_and_eval "${run_name}" "full" "${model_name}" "${model_size}" "${style}" "full" "${cfg_path}"
    done
  done

  log "Stage 2b: required zero-shot anchors"
  run_eval "4b_zeroshot" "zeroshot" "${MODEL_4B}" "4B" "zero_shot" "0" ""
  run_eval "8b_zeroshot" "zeroshot" "${MODEL_8B}" "8B" "zero_shot" "0" ""

  log "Stage 3: winner-style decision gate for 8B"
  winner_style="$(pick_winner_style)"
  log "Winner style for 8B: ${winner_style}"

  log "Stage 4: 8B Stage-B ablation (250/1000 + reuse full)"
  for sz in 250 1000; do
    run_name="8b_${sz}_${winner_style}"
    cfg_path="$(write_config "${run_name}" "${MODEL_8B}" "${winner_style}" "${SUBSET_DIR}/train_${sz}.jsonl" "full" "${USE_QLORA_8B}")"
    train_and_eval "${run_name}" "ablation" "${MODEL_8B}" "8B" "${winner_style}" "${sz}" "${cfg_path}"
  done

  if [[ "${RUN_4B_ABLATION}" == "true" ]]; then
    log "Optional: running 4B 250/1000 matched ablation for winner style"
    for sz in 250 1000; do
      run_name="4b_${sz}_${winner_style}"
      cfg_path="$(write_config "${run_name}" "${MODEL_4B}" "${winner_style}" "${SUBSET_DIR}/train_${sz}.jsonl" "full" "${USE_QLORA_4B}")"
      train_and_eval "${run_name}" "ablation" "${MODEL_4B}" "4B" "${winner_style}" "${sz}" "${cfg_path}"
    done
  else
    log "Skip optional 4B 250/1000 ablation (RUN_4B_ABLATION=false)."
  fi

  log "Stage 5: mandatory error-shift report"
  run_error_shift_report

  log "All done."
  log "Manifest: ${MANIFEST_PATH}"
  log "Error-shift markdown: ${RESULTS_ROOT}/error_shift_report.md"
}

main "$@"
