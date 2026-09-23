# ITCS 6169/8169 Assignment 1 - CNN Challenge

Reproducible PyTorch code for 16-class scene classification. The primary
classifiers are CNNs, as required by the assignment. Model selection uses only
a deterministic, class-stratified validation split. The labeled test set is
evaluated only after a final configuration has been selected.

## 1. Remote setup

```bash
cd /projects/cci-abc/HPDiL/jzhang84/assignment1
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The cci-abc base environment already has CUDA-enabled PyTorch 2.8.0 and
torchvision 0.23.0. `--system-site-packages` reuses them instead of downloading
another CUDA build.

## 2. Download and verify the data

The downloader first writes to node-local `/tmp`, verifies the expected image
layout, and then copies the complete data into the project. This avoids issuing
thousands of small network writes directly to `/projects`.

```bash
python scripts/download_data.py \
  --destination data \
  --staging /tmp/jzhang84_assignment1_data \
  --workers 12

python scripts/check_data.py --data-root data
```

Expected valid-image counts:

- `data/train`: 2,400 labeled images, 16 classes
- `data/test`: 400 labeled images, 16 classes
- `data/test2`: 400 unlabeled images

Both scripts are safe to rerun. Existing nonempty staged files are reused.

## 3. Run controlled experiments

Use the same split file (`splits/seed0_val20.json`) for every run. The training
script creates it deterministically on the first run.

Run one experiment in the foreground:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --config configs/baseline_tnet.yaml
```

Recommended experiment sequence:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/baseline_tnet.yaml
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/resnet18_linear.yaml
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/resnet18_finetune_basic.yaml
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/resnet18_finetune_strong.yaml
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/efficientnet_b0_finetune.yaml
```

Higher-capacity and domain-matched CNN experiments are also provided. The two
Places365 models are especially relevant because Places365 pretraining is for
scene recognition rather than object recognition:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/places365_resnet18_finetune.yaml
CUDA_VISIBLE_DEVICES=1 python -u train.py --config configs/places365_resnet50_finetune.yaml
CUDA_VISIBLE_DEVICES=2 python -u train.py --config configs/convnext_base_finetune.yaml
CUDA_VISIBLE_DEVICES=3 python -u train.py --config configs/regnet_y_32gf_swag_finetune.yaml
CUDA_VISIBLE_DEVICES=4 python -u train.py --config configs/resnext101_64x4d_finetune.yaml
CUDA_VISIBLE_DEVICES=5 python -u train.py --config configs/wide_resnet101_2_finetune.yaml
```

These commands may run concurrently only when the listed GPUs are otherwise
idle. Places365 weights come from the official CSAILVision Places365 release;
document this pretraining source in the report.

To keep a run alive after disconnecting:

```bash
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --config configs/resnet18_finetune_basic.yaml \
  > logs/resnet18_finetune_basic.log 2>&1 &
echo $!
tail -f logs/resnet18_finetune_basic.log
```

To use several otherwise-idle GPUs, assign a different visible GPU to each
process. Do not run two experiments on the same GPU.

Each run writes to `runs/<experiment-name>/`:

- `config.yaml`: exact resolved configuration
- `best.pt` and `last.pt`: checkpoints
- `history.csv`: epoch-level metrics
- `events.jsonl`: machine-readable log
- `curves.png`: learning curves
- `run_summary.json`: best validation result and timing

If a run directory already contains outputs, training stops rather than
silently overwriting it. Supply a new name with `--run-name` when repeating an
experiment, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --config configs/resnet18_finetune_basic.yaml \
  --run-name resnet18_finetune_basic_seed1
```

## 4. Compare validation results

```bash
python summarize_runs.py --runs-root runs --output runs/experiment_summary.csv
column -s, -t < runs/experiment_summary.csv
```

Choose the final model using validation accuracy and the training curves. Do
not use the test set to choose hyperparameters.

An ensemble can be selected on validation data without touching the test set.
For example:

```bash
python ensemble.py \
  --checkpoints \
    runs/convnext_tiny_finetune/best.pt \
    runs/regnet_y_16gf_swag_finetune/best.pt \
  --data-root dataset \
  --split val \
  --tta-horizontal \
  --output-dir runs/ensemble_convnext_regnet/validation
```

Use fixed equal weights unless a weighting rule was specified before examining
the test set. Report both component models and the ensemble rule.

After selecting a recipe, it is permissible to refit that fixed recipe on all
2,400 training images. The epoch count should come from the corresponding
validation run, not from test performance. Example:

```bash
python refit.py \
  --config configs/convnext_tiny_finetune.yaml \
  --development-checkpoint runs/convnext_tiny_finetune/best.pt \
  --run-name convnext_tiny_full_train \
  --epochs 9 \
  --data-root dataset
```

The refit checkpoint is `runs/convnext_tiny_full_train/final.pt`. Preserve the
development checkpoint and its validation accuracy as model-selection evidence.

## 5. Final labeled-test evaluation

Run this once for the selected checkpoint:

```bash
python evaluate.py \
  --checkpoint runs/resnet18_finetune_basic/best.pt \
  --data-root data \
  --output-dir runs/resnet18_finetune_basic/test_evaluation
```

This produces overall accuracy, per-class accuracy, and a confusion matrix.

## 6. Optional predictions for the unlabeled `test2` folder

```bash
python predict.py \
  --checkpoint runs/resnet18_finetune_basic/best.pt \
  --input-dir data/test2 \
  --output runs/resnet18_finetune_basic/test2_predictions.csv
```

## 7. Reproducibility notes

- Seed: 0 by default.
- Split: deterministic class-stratified 80/20 training/validation split.
- Validation receives deterministic preprocessing only; random augmentation is
  restricted to the training subset.
- Best checkpoints are selected by validation accuracy.
- Automatic mixed precision is enabled on CUDA.
- The checkpoint records the full configuration and class order.

Before submission, update `AI_USAGE.md`, fill `report/report.tex` with measured
results only, add the GitHub repository URL, and commit the split, summaries,
selected lightweight result files, and documentation. Do not commit the data or
large checkpoints; upload the final checkpoint separately and link it.
