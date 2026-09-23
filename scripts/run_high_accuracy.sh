#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/projects/cci-abc/HPDiL/jzhang84/assignment1"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

cd "$PROJECT_ROOT"
mkdir -p logs

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d dataset/train ]]; then
  echo "Dataset not found at $PROJECT_ROOT/dataset/train" >&2
  exit 1
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "GNU screen is not installed" >&2
  exit 1
fi

launch_experiment() {
  local session_name="$1"
  local gpu="$2"
  local config="$3"
  local run_name="$4"
  local log_file="$PROJECT_ROOT/logs/${session_name}.screen.log"

  if [[ -f "$PROJECT_ROOT/runs/$run_name/run_summary.json" ]]; then
    echo "Completed run $run_name already exists; skipping"
    return
  fi
  if [[ -d "$PROJECT_ROOT/runs/$run_name" ]]; then
    echo "Nonempty or partial run directory exists: runs/$run_name" >&2
    echo "Inspect it and use a new --run-name before retrying." >&2
    return
  fi
  if screen -ls | grep -q "[.]${session_name}[[:space:]]"; then
    echo "Screen $session_name already exists; skipping duplicate launch"
    return
  fi

  STY= screen -L -Logfile "$log_file" -dmS "$session_name" \
    bash -lc "cd '$PROJECT_ROOT' && CUDA_VISIBLE_DEVICES='$gpu' '$PYTHON_BIN' -u train.py --config '$config'"
  echo "Started $session_name on physical GPU $gpu: $config"
}

launch_experiment a1_places18 0 configs/places365_resnet18_finetune.yaml places365_resnet18_finetune
launch_experiment a1_places50 1 configs/places365_resnet50_finetune.yaml places365_resnet50_finetune
launch_experiment a1_convbase 2 configs/convnext_base_finetune.yaml convnext_base_finetune
launch_experiment a1_regnet32 3 configs/regnet_y_32gf_swag_finetune.yaml regnet_y_32gf_swag_finetune
launch_experiment a1_resnext101 4 configs/resnext101_64x4d_finetune.yaml resnext101_64x4d_finetune
launch_experiment a1_wideresnet101 5 configs/wide_resnet101_2_finetune.yaml wide_resnet101_2_finetune

sleep 3
screen -ls || true
echo "Monitor with: tail -f $PROJECT_ROOT/logs/a1_places50.screen.log"

