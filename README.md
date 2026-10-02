# ITCS 8169 Assignment 1 - The CNN Challenge

PyTorch code for 16-class scene classification from 2,400 training images
(the 15-Scene categories plus Flower). All classifiers are CNNs, as required by
the assignment. Model selection uses only a fixed, class-stratified validation
split; the labeled test set was evaluated once, for the selected model.

The two-page report is [`report/report.pdf`](report/report.pdf) (source:
[`report/report.tex`](report/report.tex)). How AI tools were used is
documented in [`AI_USAGE.md`](AI_USAGE.md).

## Final result

| | |
|---|---|
| Model | ResNet-50, Places365-pretrained (CSAILVision weights), all layers fine-tuned |
| Test accuracy (`test`, 400 labeled images) | **95.50%** (382/400) |
| Validation accuracy (fixed 480-image split) | **96.04%** (461/480, mixed-precision evaluation during training); 95.83% (460/480) re-evaluated in fp32, the precision used for the test result |
| Configuration | [`configs/best.yaml`](configs/best.yaml) (identical to `configs/places365_resnet50_finetune.yaml` apart from the run name) |
| Checkpoint | [`checkpoints/places365_resnet50_final.pt`](checkpoints/places365_resnet50_final.pt), 90 MiB, stored with Git LFS |
| Checkpoint SHA-256 | `1ff041ac08780582686fae80676d9113b63926c647e03bd94733bb3fd356fac4` |
| Training cost | 25 epochs (early stopping; best epoch 15), 2.8 minutes on one Quadro RTX 5000 |

Per-class test accuracy, the confusion matrix and the predictions for the
unlabeled `test2` folder are in
[`runs/places365_resnet50_finetune/`](runs/places365_resnet50_finetune/).

## 1. Environment

```bash
cd /projects/cci-abc/HPDiL/jzhang84/assignment1
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The cci-abc base environment already provides CUDA-enabled PyTorch 2.8.0 and
torchvision 0.23.0 (Python 3.13.9); `--system-site-packages` reuses them. The
exact versions, GPU and seed of every run are recorded in
`runs/<run>/environment.json`.

The checkpoint is stored with Git LFS. After cloning, fetch it with:

```bash
git lfs install
git lfs pull
```

## 2. Data

```bash
python scripts/download_data.py \
  --destination dataset \
  --staging /tmp/jzhang84_assignment1_data \
  --workers 12

python scripts/check_data.py --data-root dataset
ln -s dataset data   # some early configs use data.root: data
```

The downloader writes to node-local `/tmp` first, verifies the image layout and
only then copies the data into the project. Expected counts: `train` 2,400
images in 16 classes, `test` 400 labeled images, `test2` 400 unlabeled images.
Fifteen classes are grayscale (the 15-Scene
images, about 256 px); Flower is the only color class.

## 3. Reproduce the final model

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/best.yaml
```

This writes `runs/best_places365_resnet50/`. On the same GPU type it reproduced
the original run bit for bit: identical per-epoch metrics and identical
`best.pt` weights (best epoch 15, 96.04% validation accuracy). The ResNet-50
Places365 weights are downloaded automatically from
`http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar`
(SHA-256 `46529c86902bd0cfb0ea562a30b2850c28d2620d96282b3db9c318e1d774f6c5`).

The released checkpoint is the weights-only export of
`runs/places365_resnet50_finetune/best.pt`:

```bash
python scripts/export_checkpoint.py \
  --checkpoint runs/places365_resnet50_finetune/best.pt \
  --output checkpoints/places365_resnet50_final.pt
```

## 4. Evaluate

Labeled test set (overall accuracy, per-class accuracy, confusion matrix):

```bash
python evaluate.py \
  --checkpoint checkpoints/places365_resnet50_final.pt \
  --data-root dataset \
  --output-dir runs/final_test_evaluation
```

Expected output: `accuracy 0.955, correct 382, total 400`.

Predictions for the unlabeled `test2` folder:

```bash
python predict.py \
  --checkpoint checkpoints/places365_resnet50_final.pt \
  --input-dir dataset/test2 \
  --output runs/final_test2_predictions.csv
```

## 5. Experiments

Every run uses the same split, `splits/seed0_val20.json` (1,920 training and
480 validation images, 30 per class). Each run directory under `runs/` holds the
resolved `config.yaml`, `environment.json`, per-epoch `history.csv` and
`events.jsonl`, and `run_summary.json`; checkpoints are not committed. A run is
started with

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/<config>.yaml [--run-name <name>] [--seed <n>]
```

`--seed` changes the training seed but keeps the validation split fixed.
`train.py` refuses to overwrite a non-empty run directory.

| Run directory | Config | Val. acc. (%) |
|---|---|---|
| `baseline_tnet_run3` | `baseline_tnet.yaml` (starter TNet) | 45.42 |
| `resnet18_scratch` | `resnet18_scratch.yaml` | 80.42 |
| `resnet18_linear_run3` | `resnet18_linear.yaml` | 80.21 |
| `resnet18_finetune_basic_run3` | `resnet18_finetune_basic.yaml` | 90.42 |
| `resnet18_finetune_strong_run3` | `resnet18_finetune_strong.yaml` | 90.83 |
| `resnet50_imagenet_finetune` | `resnet50_imagenet_finetune.yaml` | 92.08 |
| `places365_resnet18_finetune` | `places365_resnet18_finetune.yaml` | 94.17 |
| `places365_resnet50_strong` | `places365_resnet50_strong.yaml` | 94.79 |
| `places365_resnet50_finetune` (final) | `places365_resnet50_finetune.yaml` | **96.04** |
| `places365_resnet50_finetune_seed1`, `_seed2` | `places365_resnet50_finetune.yaml --seed 1/2` | 95.42, 95.83 |
| `best_places365_resnet50` (reproduction) | `best.yaml` | 96.04 |

`runs/experiment_summary.csv` collects all of them
(`python summarize_runs.py --runs-root runs`). `scripts/run_pipeline.sh`
(starter TNet and ImageNet ResNet-18 runs), `scripts/run_followup.sh` (the
ResNet-50 pretraining comparison and the random-init ResNet-18) and
`scripts/run_resnet_checks.sh` (seeds, strong augmentation and ResNet-18 on the
final recipe) launch runs in GNU Screen sessions; the exact configuration of
every run is its `runs/<run>/config.yaml`.

### Analysis files

`runs/analysis/` holds the fp32 validation predictions of the two ResNet-50
runs used for the paired McNemar test in the report.
`runs/model_selection/places365_r50_tta/` shows that horizontal-flip test-time
augmentation did not help the final model on validation (95.63% vs. 95.83%
without it, both evaluated in fp32), so the final model does not use it.

### Report figure

```bash
python scripts/make_figures.py --runs-root runs --output report/figures/curves.pdf
```

## Repository layout

```
train.py              training with a fixed validation split and best-checkpoint selection
evaluate.py           labeled-split evaluation: accuracy, per-class accuracy, confusion matrix
predict.py            predictions for an unlabeled folder (test2)
ensemble.py           probability ensembles and flip TTA on val/test/test2
refit.py              optional refit of a fixed recipe on all 2,400 images (not used for the final model)
summarize_runs.py     collects run summaries into one CSV
src/                  data pipeline, model builders, utilities
configs/              one YAML file per experiment; best.yaml is the final recipe
splits/               the saved stratified validation split
runs/                 lightweight artifacts of every run (no checkpoints)
checkpoints/          the released final checkpoint (Git LFS)
scripts/              data download/check, experiment launchers, figures, checkpoint export
report/               report source, figure and PDF
```
