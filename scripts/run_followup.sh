#!/usr/bin/env bash
# Follow-up controlled experiments run after the first model sweep. Each
# screen session runs its jobs one after another on a single GPU.
set -euo pipefail

PROJECT_ROOT="/projects/cci-abc/HPDiL/jzhang84/assignment1"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

cd "$PROJECT_ROOT"
mkdir -p logs

if ! command -v screen >/dev/null 2>&1; then
  echo "GNU screen is not installed" >&2
  exit 1
fi

launch_queue() {
  local session_name="$1"
  local gpu="$2"
  shift 2
  local log_file="$PROJECT_ROOT/logs/${session_name}.screen.log"

  if screen -ls | grep -q "[.]${session_name}[[:space:]]"; then
    echo "Screen $session_name already exists; skipping duplicate launch"
    return
  fi

  local commands=""
  local job
  for job in "$@"; do
    commands+="CUDA_VISIBLE_DEVICES='$gpu' '$PYTHON_BIN' -u train.py $job; "
  done
  STY= screen -L -Logfile "$log_file" -dmS "$session_name" \
    bash -lc "cd '$PROJECT_ROOT' && $commands"
  echo "Started $session_name on physical GPU $gpu"
}

# Pretraining domain (identical ResNet-50 recipe, Places365 vs ImageNet
# weights) and the value of pretraining at all (random-init ResNet-18).
launch_queue a1_followup_gpu0 0 "--config configs/places365_resnet50_finetune.yaml"
launch_queue a1_followup_gpu1 1 "--config configs/resnet50_imagenet_finetune.yaml"
launch_queue a1_followup_gpu2 2 "--config configs/resnet18_scratch.yaml"

sleep 3
screen -ls || true
