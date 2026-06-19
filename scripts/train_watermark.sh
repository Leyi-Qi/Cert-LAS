#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/train_watermark.json}"
if [[ $# -ge 1 ]]; then
  CONFIG="$1"
  shift
fi

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29511}"

GPU_ARGS=()
if [[ -n "${GPU_IDS:-}" ]]; then
  GPU_ARGS=(--gpu_ids "${GPU_IDS}")
fi

"${ACCELERATE_BIN}" launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${GPU_ARGS[@]}" \
  -m cert_las.train.watermark \
  --config_path "${CONFIG}" \
  "$@"
