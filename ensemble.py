#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from evaluate import save_confusion_plot
from src.data import FlatImageDataset, build_transforms
from src.models import build_model, clone_model_config_without_pretraining
from src.utils import resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an equal- or fixed-weight CNN probability ensemble"
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument(
        "--split",
        choices=("val", "test", "test2"),
        default="val",
        help="val/test are labeled; test2 is the unlabeled flat folder",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tta-horizontal", action="store_true")
    parser.add_argument("--device")
    return parser.parse_args()


def make_loader(config: dict, data_root: str, split: str):
    data_config = dict(config["data"])
    data_config["root"] = data_root
    transform = build_transforms(data_config, training=False)
    classes = None
    if split == "val":
        root = Path(data_root) / str(data_config.get("train_dir", "train"))
        dataset = datasets.ImageFolder(root, transform=transform)
        with Path(data_config["split_file"]).open("r", encoding="utf-8") as stream:
            split_definition = json.load(stream)
        indices = list(split_definition["val_indices"])
        filenames = [str(Path(dataset.samples[index][0]).relative_to(root)) for index in indices]
        selected = Subset(dataset, indices)
        classes = dataset.classes
    elif split == "test":
        root = Path(data_root) / str(data_config.get("test_dir", "test"))
        dataset = datasets.ImageFolder(root, transform=transform)
        filenames = [str(Path(path).relative_to(root)) for path, _ in dataset.samples]
        selected = dataset
        classes = dataset.classes
    else:
        # Unlabeled flat folder: the loader yields file names instead of labels.
        dataset = FlatImageDataset(Path(data_root) / split, transform=transform)
        filenames = [path.name for path in dataset.paths]
        selected = dataset
    loader = DataLoader(
        selected,
        batch_size=max(1, min(16, int(data_config["batch_size"]))),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )
    return loader, classes, filenames


@torch.inference_mode()
def predict_one(checkpoint_path: str, data_root: str, split: str, device, use_tta: bool):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    loader, dataset_classes, filenames = make_loader(config, data_root, split)
    class_names = list(checkpoint["class_names"])
    if dataset_classes is not None and dataset_classes != class_names:
        raise RuntimeError(
            f"Class order mismatch for {checkpoint_path}: "
            f"checkpoint={class_names}, dataset={dataset_classes}"
        )
    model, _ = build_model(
        clone_model_config_without_pretraining(config["model"]),
        num_classes=len(class_names),
        image_size=int(config["data"]["image_size"]),
        load_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    probabilities = []
    labels = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        current = model(images).softmax(dim=1)
        if use_tta:
            current = (current + model(torch.flip(images, dims=[3])).softmax(dim=1)) / 2
        probabilities.append(current.cpu())
        if dataset_classes is not None:
            labels.append(targets)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metadata = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "model": config["model"]["name"],
        "image_size": int(config["data"]["image_size"]),
        "epoch": checkpoint.get("epoch"),
        "best_val_accuracy": checkpoint.get("best_val_accuracy"),
    }
    return (
        torch.cat(probabilities),
        torch.cat(labels) if labels else None,
        class_names,
        filenames,
        metadata,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    weights = args.weights or [1.0] * len(args.checkpoints)
    if len(weights) != len(args.checkpoints) or any(weight <= 0 for weight in weights):
        raise ValueError("--weights must provide one positive value per checkpoint")
    weight_tensor = torch.tensor(weights, dtype=torch.float64)
    weight_tensor /= weight_tensor.sum()

    model_probabilities = []
    labels = None
    class_names = None
    filenames = None
    individual = []
    for path in args.checkpoints:
        probabilities, current_labels, current_classes, current_filenames, metadata = (
            predict_one(path, args.data_root, args.split, device, args.tta_horizontal)
        )
        if labels is not None and not torch.equal(labels, current_labels):
            raise RuntimeError("Label order differs between checkpoints")
        if class_names is not None and class_names != current_classes:
            raise RuntimeError("Class order differs between checkpoints")
        if filenames is not None and filenames != current_filenames:
            raise RuntimeError("Filename order differs between checkpoints")
        labels = current_labels
        class_names = current_classes
        filenames = current_filenames
        model_probabilities.append(probabilities)
        if labels is not None:
            metadata["accuracy"] = float((probabilities.argmax(1) == labels).float().mean())
        individual.append(metadata)

    assert class_names is not None and filenames is not None
    stacked = torch.stack(model_probabilities).to(torch.float64)
    ensemble_probabilities = (stacked * weight_tensor[:, None, None]).sum(dim=0)
    predictions = ensemble_probabilities.argmax(dim=1)
    confidence = ensemble_probabilities.max(dim=1).values

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "split": args.split,
        "total": len(predictions),
        "tta_horizontal": args.tta_horizontal,
        "normalized_weights": weight_tensor.tolist(),
        "individual_models": individual,
    }
    if labels is not None:
        num_classes = len(class_names)
        flat_indices = labels * num_classes + predictions
        confusion = torch.bincount(flat_indices, minlength=num_classes**2).reshape(
            num_classes, num_classes
        ).numpy()
        metrics["accuracy"] = float((predictions == labels).float().mean())
        metrics["correct"] = int((predictions == labels).sum())
        np.savetxt(output_dir / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
        save_confusion_plot(confusion, class_names, output_dir / "confusion_matrix.png")
        with (output_dir / "per_class_accuracy.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=("class", "correct", "total", "accuracy"))
            writer.writeheader()
            for index, name in enumerate(class_names):
                class_total = int(confusion[index].sum())
                writer.writerow(
                    {
                        "class": name,
                        "correct": int(confusion[index, index]),
                        "total": class_total,
                        "accuracy": confusion[index, index] / class_total if class_total else 0.0,
                    }
                )
    save_json(metrics, output_dir / "metrics.json")

    fieldnames = ["filename", "predicted_class", "confidence"]
    if labels is not None:
        fieldnames += ["true_class", "correct"]
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, (filename, prediction, score) in enumerate(
            zip(filenames, predictions, confidence)
        ):
            row = {
                "filename": filename,
                "predicted_class": class_names[int(prediction)],
                "confidence": float(score),
            }
            if labels is not None:
                row["true_class"] = class_names[int(labels[index])]
                row["correct"] = int(labels[index] == prediction)
            writer.writerow(row)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
