#!/usr/bin/env bash
set -euo pipefail

# Background supervisor for Stage-1 train/eval matrix.
# Features:
# - start/stop/status lifecycle
# - main run process management
# - watchdog process (stalled-log alert)
# - ntfy monitor process (stage/progress/finish notifications)
#
# Usage:
#   bash stage1/scripts/run_train_eval_background.sh start
#   bash stage1/scripts/run_train_eval_background.sh status
#   bash stage1/scripts/run_train_eval_background.sh stop
#
# Common env vars:
#   RUN_SCRIPT=stage1/scripts/run_train_eval_matrix.sh
#   LOG_DIR=stage1/logs
#   RUN_LOG=stage1/logs/train_eval_matrix_full.log
#   NTFY_TOPIC=finqa-stage1
#   NTFY_SERVER=https://ntfy.sh
#   ENABLE_NTFY=true
#   WATCHDOG_TIMEOUT=1800
#   WATCHDOG_CHECK_INTERVAL=60

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_SCRIPT="${RUN_SCRIPT:-${STAGE1_ROOT}/scripts/run_train_eval_matrix.sh}"
LOG_DIR="${LOG_DIR:-${STAGE1_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/train_eval_matrix_full.log}"
SUP_LOG="${SUP_LOG:-${LOG_DIR}/train_eval_background.log}"
EXIT_CODE_FILE="${EXIT_CODE_FILE:-${LOG_DIR}/train_eval_exit_code.txt}"

MAIN_PID_FILE="${MAIN_PID_FILE:-${LOG_DIR}/train_eval_main.pid}"
WATCHDOG_PID_FILE="${WATCHDOG_PID_FILE:-${LOG_DIR}/train_eval_watchdog.pid}"
NTFY_PID_FILE="${NTFY_PID_FILE:-${LOG_DIR}/train_eval_ntfy.pid}"

NTFY_TOPIC="${NTFY_TOPIC:-finqa-stage1}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
ENABLE_NTFY="${ENABLE_NTFY:-true}"
NOTIFY_EVERY="${NOTIFY_EVERY:-100}"

WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-1800}"
WATCHDOG_CHECK_INTERVAL="${WATCHDOG_CHECK_INTERVAL:-60}"

mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] [bg-runner] $*" | tee -a "${SUP_LOG}"; }

send_ntfy() {
  local title="$1"
  local msg="$2"
  local priority="${3:-default}"

  if [[ "${ENABLE_NTFY}" != "true" ]]; then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    return 0
  fi
  curl -s -o /dev/null \
    -H "Title: ${title}" \
    -H "Priority: ${priority}" \
    -d "${msg}" \
    "${NTFY_SERVER}/${NTFY_TOPIC}" || true
}

pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

read_pid() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]]; then
    tr -d '[:space:]' < "${pid_file}"
  else
    echo ""
  fi
}

ensure_not_running() {
  local main_pid
  main_pid="$(read_pid "${MAIN_PID_FILE}")"
  if [[ -n "${main_pid}" ]] && pid_alive "${main_pid}"; then
    log "Main process is already running (pid=${main_pid})."
    exit 1
  fi
}

start_main() {
  : > "${SUP_LOG}"
  : > "${RUN_LOG}"
  rm -f "${EXIT_CODE_FILE}"

  log "Starting main run script (setsid): ${RUN_SCRIPT}"
  setsid bash -lc \
    "cd '${STAGE1_ROOT}' && bash '${RUN_SCRIPT}' >> '${RUN_LOG}' 2>&1; code=\$?; echo \"\$code\" > '${EXIT_CODE_FILE}'; exit \$code" \
    </dev/null >/dev/null 2>&1 &
  local main_pid=$!
  echo "${main_pid}" > "${MAIN_PID_FILE}"
  log "Main started (pid=${main_pid}), log=${RUN_LOG}"
  send_ntfy "Stage1 Run Started" "Main PID=${main_pid}" "low"
}

_watchdog_loop() {
  local main_pid="$1"
  local last_alert_ts=0
  while true; do
    sleep "${WATCHDOG_CHECK_INTERVAL}"
    if ! pid_alive "${main_pid}"; then
      break
    fi
    if [[ ! -f "${RUN_LOG}" ]]; then
      continue
    fi

    now="$(date +%s)"
    last_mod="$(stat -c %Y "${RUN_LOG}" 2>/dev/null || echo "${now}")"
    elapsed=$(( now - last_mod ))

    if (( elapsed >= WATCHDOG_TIMEOUT && now - last_alert_ts >= WATCHDOG_TIMEOUT )); then
      mins=$(( elapsed / 60 ))
      echo "[$(ts)] [watchdog] no log update for ${mins} min" >> "${SUP_LOG}"
      send_ntfy "Stage1 Watchdog" "No log update for ${mins} min. Please check run state." "urgent"
      last_alert_ts="${now}"
    fi
  done
}

start_watchdog() {
  local main_pid
  main_pid="$(read_pid "${MAIN_PID_FILE}")"
  [[ -n "${main_pid}" ]] || { log "Cannot start watchdog: missing main pid."; return 1; }

  setsid bash -lc \
    "cd '${STAGE1_ROOT}' && bash '${SCRIPT_DIR}/run_train_eval_background.sh' _watchdog_loop '${main_pid}'" \
    </dev/null >/dev/null 2>&1 &
  local watchdog_pid=$!
  echo "${watchdog_pid}" > "${WATCHDOG_PID_FILE}"
  log "Watchdog started (pid=${watchdog_pid}, timeout=${WATCHDOG_TIMEOUT}s, setsid=true)"
}

_ntfy_loop() {
  local main_pid="$1"
  local current_stage=""
  local current_run=""
  local last_notified=0

  send_ntfy "Stage1 Monitor" "ntfy monitor started. topic=${NTFY_TOPIC}" "low"

  tail -n 0 -F "${RUN_LOG}" 2>/dev/null | tr '\r' '\n' | while IFS= read -r line; do
    if ! pid_alive "${main_pid}"; then
      send_ntfy "Stage1 Monitor" "main process ended; ntfy monitor exiting." "low"
      break
    fi

    # Stage transitions from run_train_eval_matrix.sh
    if [[ "${line}" =~ Stage[[:space:]][0-9]+: ]]; then
      stage_line="${line#*] }"
      if [[ "${stage_line}" != "${current_stage}" ]]; then
        current_stage="${stage_line}"
        send_ntfy "Stage1 Stage Update" "${current_stage}" "default"
      fi
    fi

    if [[ "${line}" =~ Winner[[:space:]]style[[:space:]]for[[:space:]]8B:[[:space:]](.+) ]]; then
      winner="${BASH_REMATCH[1]}"
      send_ntfy "Stage1 Decision Gate" "8B winner_style=${winner}" "high"
    fi

    # Eval run marker from run_train_eval_matrix.sh
    if [[ "${line}" =~ Eval[[:space:]]run=([^[:space:]]+) ]]; then
      current_run="${BASH_REMATCH[1]}"
      last_notified=0
      send_ntfy "Stage1 Eval Started" "${current_run}" "low"
    fi

    # tqdm lines from eval_finqa.py
    if [[ "${line}" =~ Evaluating[[:space:]]FinQA:.*\|\ *([0-9]+)/([0-9]+)\ \[ ]]; then
      cur="${BASH_REMATCH[1]}"
      total="${BASH_REMATCH[2]}"
      if (( cur - last_notified >= NOTIFY_EVERY )) || (( cur > 0 && cur == total )); then
        pct=$(( cur * 100 / total ))
        send_ntfy "Stage1 Eval Progress" "${current_run:-eval}: ${cur}/${total} (${pct}%)"
        last_notified="${cur}"
      fi
    fi

    # Save markers from eval_finqa.py
    if [[ "${line}" =~ \[saved\][[:space:]](.*/summary\.json) ]]; then
      saved_file="${BASH_REMATCH[1]}"
      send_ntfy "Stage1 Eval Saved" "${current_run:-eval} -> ${saved_file}" "default"
    fi

    # Failure keywords
    if [[ "${line}" =~ (Traceback|ERROR|Error:|RuntimeError|Exception) ]]; then
      send_ntfy "Stage1 Alert" "Possible error detected in log: ${line}" "high"
    fi

    # Completion marker
    if [[ "${line}" =~ All[[:space:]]done\. ]]; then
      send_ntfy "Stage1 Run Completed" "Main workflow finished." "high"
    fi
  done
}

start_ntfy_monitor() {
  local main_pid
  main_pid="$(read_pid "${MAIN_PID_FILE}")"
  [[ -n "${main_pid}" ]] || { log "Cannot start ntfy monitor: missing main pid."; return 1; }

  setsid bash -lc \
    "cd '${STAGE1_ROOT}' && bash '${SCRIPT_DIR}/run_train_eval_background.sh' _ntfy_loop '${main_pid}'" \
    </dev/null >/dev/null 2>&1 &
  local ntfy_pid=$!
  echo "${ntfy_pid}" > "${NTFY_PID_FILE}"
  log "ntfy monitor started (pid=${ntfy_pid}, topic=${NTFY_SERVER}/${NTFY_TOPIC}, setsid=true)"
}

stop_one() {
  local pid_file="$1"
  local name="$2"
  local pid
  pid="$(read_pid "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if pid_alive "${pid}"; then
    kill "${pid}" 2>/dev/null || true
    sleep 1
    if pid_alive "${pid}"; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
    log "Stopped ${name} (pid=${pid})"
  fi
  rm -f "${pid_file}"
}

cmd_start() {
  ensure_not_running
  start_main
  start_watchdog
  start_ntfy_monitor
  log "Background workflow started."
}

cmd_stop() {
  stop_one "${NTFY_PID_FILE}" "ntfy monitor"
  stop_one "${WATCHDOG_PID_FILE}" "watchdog"
  stop_one "${MAIN_PID_FILE}" "main run"
  send_ntfy "Stage1 Run Stopped" "Background processes stopped by user." "high"
  log "All processes stopped."
}

cmd_status() {
  local main_pid watchdog_pid ntfy_pid
  main_pid="$(read_pid "${MAIN_PID_FILE}")"
  watchdog_pid="$(read_pid "${WATCHDOG_PID_FILE}")"
  ntfy_pid="$(read_pid "${NTFY_PID_FILE}")"

  echo "RUN_SCRIPT=${RUN_SCRIPT}"
  echo "RUN_LOG=${RUN_LOG}"
  echo "SUP_LOG=${SUP_LOG}"
  echo "NTFY=${ENABLE_NTFY} topic=${NTFY_SERVER}/${NTFY_TOPIC}"
  echo

  if [[ -n "${main_pid}" ]] && pid_alive "${main_pid}"; then
    echo "main     : running (pid=${main_pid})"
  else
    echo "main     : stopped"
  fi
  if [[ -n "${watchdog_pid}" ]] && pid_alive "${watchdog_pid}"; then
    echo "watchdog : running (pid=${watchdog_pid})"
  else
    echo "watchdog : stopped"
  fi
  if [[ -n "${ntfy_pid}" ]] && pid_alive "${ntfy_pid}"; then
    echo "ntfy     : running (pid=${ntfy_pid})"
  else
    echo "ntfy     : stopped"
  fi

  if [[ -f "${EXIT_CODE_FILE}" ]]; then
    code="$(tr -d '[:space:]' < "${EXIT_CODE_FILE}")"
    echo "last_exit: ${code}"
  fi
}

usage() {
  cat <<EOF
Usage: bash stage1/scripts/run_train_eval_background.sh <start|stop|status|restart>
EOF
}

main() {
  action="${1:-status}"
  case "${action}" in
    _watchdog_loop)
      _watchdog_loop "${2:-}"
      ;;
    _ntfy_loop)
      _ntfy_loop "${2:-}"
      ;;
    start) cmd_start ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    restart)
      cmd_stop
      sleep 1
      cmd_start
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
