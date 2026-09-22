#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import FlatImageDataset, build_transforms
from src.models import build_model, clone_model_config_without_pretraining
from src.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict labels for a flat image folder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    class_names = list(checkpoint["class_names"])
    model, _ = build_model(
        clone_model_config_without_pretraining(config["model"]),
        num_classes=len(class_names),
        image_size=int(config["data"]["image_size"]),
        load_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    dataset = FlatImageDataset(
        args.input_dir, transform=build_transforms(config["data"], training=False)
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    rows = []
    with torch.inference_mode():
        for images, filenames in loader:
            probabilities = model(images.to(device, non_blocking=True)).softmax(dim=1)
            confidence, indices = probabilities.max(dim=1)
            for filename, index, score in zip(filenames, indices.cpu(), confidence.cpu()):
                class_index = int(index)
                rows.append(
                    {
                        "filename": filename,
                        "predicted_index": class_index,
                        "predicted_class": class_names[class_index],
                        "confidence": float(score),
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {output.resolve()}")


if __name__ == "__main__":
    main()

