#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

VSR_CONFIG="${VSR_CONFIG:-configs/eval_vsr.json}"
RADIUS_CONFIG="${RADIUS_CONFIG:-configs/eval_radius.json}"
if [[ $# -ge 1 && "$1" != --* ]]; then
  VSR_CONFIG="$1"
  shift
fi
if [[ $# -ge 1 && "$1" != --* ]]; then
  RADIUS_CONFIG="$1"
  shift
fi

VSR_ARGS=("$@")

PYTHON="${PYTHON:-python}"

VSR_OUTPUT_DIR="$("${PYTHON}" - "${VSR_CONFIG}" "${VSR_ARGS[@]}" <<'PY'
import sys

from cert_las.eval.vsr import build_parser
from cert_las.utils.config import parse_args_with_config

args = parse_args_with_config(build_parser(), ["--config_path", sys.argv[1], *sys.argv[2:]])
print(args.output_dir)
PY
)"

"${PYTHON}" -m cert_las.eval.vsr \
  --config_path "${VSR_CONFIG}" \
  "${VSR_ARGS[@]}"

PER_TRIAL_CSV="$("${PYTHON}" - "${VSR_OUTPUT_DIR}" <<'PY'
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
files = sorted(output_dir.glob("*_per_trial.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
if not files:
    raise SystemExit(f"No *_per_trial.csv found in {output_dir}")
print(files[0])
PY
)"

"${PYTHON}" -m cert_las.eval.radius \
  --config_path "${RADIUS_CONFIG}" \
  --input_csv "${PER_TRIAL_CSV}"
