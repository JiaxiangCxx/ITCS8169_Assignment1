#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/projects/cci-abc/HPDiL/jzhang84/assignment1"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

cd "$PROJECT_ROOT"
mkdir -p logs

echo "[$(date -Is)] Resuming dataset download"
"$PYTHON_BIN" scripts/download_data.py \
  --destination data \
  --staging /tmp/jzhang84_assignment1_data \
  --workers 2

echo "[$(date -Is)] Validating dataset"
"$PYTHON_BIN" scripts/check_data.py --data-root data

echo "[$(date -Is)] Creating the shared stratified split"
"$PYTHON_BIN" -c 'from src.data import make_train_val_loaders; from src.utils import load_config; c=load_config("configs/baseline_tnet.yaml"); _,_,names,info=make_train_val_loaders(c["data"], int(c["experiment"]["seed"])); print(names); print(info)'

echo "[$(date -Is)] Downloading pretrained CNN weights once"
"$PYTHON_BIN" -c 'from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.DEFAULT); print("Pretrained weights ready")'

launch_experiment() {
  local session_name="$1"
  local gpu="$2"
  local config="$3"
  local log_file="$PROJECT_ROOT/logs/${session_name}.screen.log"

  if screen -ls | grep -q "[.]${session_name}[[:space:]]"; then
    echo "Screen ${session_name} already exists; skipping duplicate launch"
    return
  fi

  STY= screen -L -Logfile "$log_file" -dmS "$session_name" \
    bash -lc "cd '$PROJECT_ROOT' && CUDA_VISIBLE_DEVICES='$gpu' '$PYTHON_BIN' -u train.py --config '$config'"
  echo "Started ${session_name} on physical GPU ${gpu} with ${config}"
}

echo "[$(date -Is)] Launching experiments"
launch_experiment a1_baseline 0 configs/baseline_tnet.yaml
launch_experiment a1_linear 1 configs/resnet18_linear.yaml
launch_experiment a1_basic 2 configs/resnet18_finetune_basic.yaml
launch_experiment a1_strong 3 configs/resnet18_finetune_strong.yaml

sleep 5
echo "[$(date -Is)] Active screens"
screen -ls
echo "[$(date -Is)] Pipeline launch complete"
