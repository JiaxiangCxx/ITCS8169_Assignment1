#!/usr/bin/env python3
"""Refit a validation-selected CNN recipe on all 2,400 training images."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.data import build_transforms
from src.models import build_model, parameter_counts, set_backbone_trainable
from src.utils import (
    append_jsonl,
    atomic_torch_save,
    load_config,
    plot_history,
    resolve_device,
    save_json,
    save_yaml,
    seed_worker,
    set_seed,
    write_history_csv,
)
from train import build_optimizer, build_scheduler, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit a selected recipe using the full training set"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--development-checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    development = torch.load(
        args.development_checkpoint, map_location="cpu", weights_only=False
    )
    config = load_config(args.config)
    config["experiment"]["name"] = args.run_name
    config["data"]["root"] = args.data_root
    epochs = int(args.epochs or development["epoch"])
    config["train"]["epochs"] = epochs

    run_dir = Path(args.runs_root) / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config.yaml")

    seed = int(config["experiment"].get("seed", 0))
    set_seed(seed)
    device = resolve_device(args.device)
    amp_enabled = bool(config["train"].get("amp", True)) and device.type == "cuda"
    train_root = Path(args.data_root) / str(config["data"].get("train_dir", "train"))
    dataset = datasets.ImageFolder(
        train_root, transform=build_transforms(config["data"], training=True)
    )
    if dataset.classes != list(development["class_names"]):
        raise RuntimeError("Development checkpoint and full dataset class orders differ")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["data"].get("num_workers", 4)) > 0,
        worker_init_fn=seed_worker,
    )

    model, head_parameters = build_model(
        config["model"], len(dataset.classes), int(config["data"]["image_size"])
    )
    freeze_epochs = int(config["model"].get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0:
        set_backbone_trainable(model, head_parameters, trainable=False)
    model.to(device)
    optimizer = build_optimizer(model, head_parameters, config["train"])
    scheduler = build_scheduler(optimizer, config["train"])
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["train"].get("label_smoothing", 0.0))
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            set_backbone_trainable(model, head_parameters, trainable=True)
        epoch_start = time.time()
        loss, accuracy = train_one_epoch(
            model, loader, criterion, optimizer, scaler, device, amp_enabled
        )
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "train_accuracy": accuracy,
            "backbone_lr": learning_rates[0],
            "classifier_lr": learning_rates[1],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        append_jsonl(row, run_dir / "events.jsonl")
        write_history_csv(history, run_dir / "history.csv")
        print(
            f"Epoch {epoch:02d}/{epochs} | loss {loss:.4f} | "
            f"training accuracy {accuracy:.4f} | {row['epoch_seconds']:.1f}s",
            flush=True,
        )

    payload = {
        "epoch": epochs,
        "model_state": model.state_dict(),
        "config": config,
        "class_names": dataset.classes,
        "full_train_size": len(dataset),
        "selection_checkpoint": str(Path(args.development_checkpoint).resolve()),
        "selection_val_accuracy": float(development["best_val_accuracy"]),
        "torch_version": torch.__version__,
    }
    atomic_torch_save(payload, run_dir / "final.pt")
    total_parameters, trainable_parameters = parameter_counts(model)
    summary = {
        "experiment": args.run_name,
        "model": config["model"]["name"],
        "epochs": epochs,
        "full_train_size": len(dataset),
        "selection_checkpoint": str(Path(args.development_checkpoint).resolve()),
        "selection_val_accuracy": float(development["best_val_accuracy"]),
        "elapsed_seconds": time.time() - start,
        "parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "python": sys.version,
        "platform": platform.platform(),
    }
    save_json(summary, run_dir / "refit_summary.json")
    print(f"Full-data checkpoint: {(run_dir / 'final.pt').resolve()}")


if __name__ == "__main__":
    main()

