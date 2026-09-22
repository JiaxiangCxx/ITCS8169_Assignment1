#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data import make_labeled_loader
from src.models import build_model, clone_model_config_without_pretraining
from src.utils import resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one selected checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device")
    return parser.parse_args()


@torch.inference_mode()
def collect_predictions(model, loader, device, num_classes: int):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        indices = labels.cpu() * num_classes + predictions.cpu()
        confusion += torch.bincount(indices, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )
        total_examples += labels.size(0)
    return total_loss / total_examples, confusion.numpy()


def save_confusion_plot(matrix: np.ndarray, class_names: list[str], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping confusion-matrix plot")
        return
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=60, ha="right")
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set(xlabel="Predicted class", ylabel="True class", title="Confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    config["data"]["root"] = args.data_root
    class_names = list(checkpoint["class_names"])
    loader, dataset_classes = make_labeled_loader(config["data"], split=args.split)
    if dataset_classes != class_names:
        raise RuntimeError(
            f"Checkpoint class order {class_names} does not match dataset {dataset_classes}"
        )

    model, _ = build_model(
        clone_model_config_without_pretraining(config["model"]),
        num_classes=len(class_names),
        image_size=int(config["data"]["image_size"]),
        load_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    loss, confusion = collect_predictions(model, loader, device, len(class_names))
    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    accuracy = correct / total

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    per_class_rows = []
    for index, name in enumerate(class_names):
        class_total = int(confusion[index].sum())
        class_correct = int(confusion[index, index])
        per_class_rows.append(
            {
                "class": name,
                "correct": class_correct,
                "total": class_total,
                "accuracy": class_correct / class_total if class_total else 0.0,
            }
        )
    with (output_dir / "per_class_accuracy.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)

    metrics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "loss": loss,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "checkpoint_best_val_accuracy": float(checkpoint["best_val_accuracy"]),
    }
    save_json(metrics, output_dir / "metrics.json")
    save_confusion_plot(confusion, class_names, output_dir / "confusion_matrix.png")
    print(metrics)


if __name__ == "__main__":
    main()

