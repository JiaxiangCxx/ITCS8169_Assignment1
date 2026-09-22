#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import make_train_val_loaders
from src.models import build_model, parameter_counts, set_backbone_trainable
from src.utils import (
    append_jsonl,
    atomic_torch_save,
    load_config,
    plot_history,
    resolve_device,
    save_json,
    save_yaml,
    set_seed,
    write_history_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN for Assignment 1")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--run-name", help="Override experiment.name")
    parser.add_argument("--data-root", help="Override data.root")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--device", help="For example cuda, cuda:0, or cpu")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
    return total_loss / total_examples, total_correct / total_examples


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
    return total_loss / total_examples, total_correct / total_examples


def build_optimizer(model, head_parameters, train_config: dict):
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [p for p in model.parameters() if id(p) not in head_ids]
    base_lr = float(train_config["learning_rate"])
    head_lr = base_lr * float(train_config.get("head_lr_multiplier", 1.0))
    groups = [
        {"params": backbone_parameters, "lr": base_lr, "name": "backbone"},
        {"params": head_parameters, "lr": head_lr, "name": "classifier"},
    ]
    name = str(train_config.get("optimizer", "adamw")).lower()
    weight_decay = float(train_config.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(groups, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            groups,
            momentum=float(train_config.get("momentum", 0.9)),
            nesterov=True,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, train_config: dict):
    name = str(train_config.get("scheduler", "none")).lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_config["epochs"]),
            eta_min=float(train_config.get("minimum_learning_rate", 1e-7)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_accuracy: float,
    config: dict,
    class_names: list[str],
    split_info: dict,
) -> dict:
    return {
        "epoch": epoch,
        "best_val_accuracy": best_val_accuracy,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "config": config,
        "class_names": class_names,
        "split_info": split_info,
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.run_name:
        config["experiment"]["name"] = args.run_name
    if args.data_root:
        config["data"]["root"] = args.data_root

    run_name = str(config["experiment"]["name"])
    run_dir = Path(args.runs_root) / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Use --run-name to avoid overwriting."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config.yaml")

    seed = int(config["experiment"].get("seed", 0))
    set_seed(seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp_enabled = bool(config["train"].get("amp", True)) and device.type == "cuda"

    train_loader, val_loader, class_names, split_info = make_train_val_loaders(
        config["data"], seed=seed
    )
    model, head_parameters = build_model(
        config["model"],
        num_classes=len(class_names),
        image_size=int(config["data"]["image_size"]),
    )
    freeze_epochs = int(config["model"].get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0:
        set_backbone_trainable(model, head_parameters, trainable=False)
    model = model.to(device)

    trainable_at_start = parameter_counts(model)
    optimizer = build_optimizer(model, head_parameters, config["train"])
    scheduler = build_scheduler(optimizer, config["train"])
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["train"].get("label_smoothing", 0.0))
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "amp": amp_enabled,
        "class_names": class_names,
        "parameter_count": trainable_at_start[0],
        "trainable_parameter_count_at_start": trainable_at_start[1],
        **split_info,
    }
    save_json(environment, run_dir / "environment.json")
    print(environment)

    epochs = int(config["train"]["epochs"])
    patience = int(config["train"].get("early_stopping_patience", 0))
    best_val_accuracy = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            set_backbone_trainable(model, head_parameters, trainable=True)
            print(f"Epoch {epoch}: unfroze the CNN backbone")

        epoch_start = time.time()
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
        )
        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device, amp_enabled
        )
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        if scheduler is not None:
            scheduler.step()

        improved = val_accuracy > best_val_accuracy
        if improved:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "backbone_lr": current_lrs[0],
            "classifier_lr": current_lrs[1],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        append_jsonl(row, run_dir / "events.jsonl")
        write_history_csv(history, run_dir / "history.csv")

        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_accuracy,
            config,
            class_names,
            split_info,
        )
        atomic_torch_save(payload, run_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, run_dir / "best.pt")

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.4f} | "
            f"best {best_val_accuracy:.4f} (epoch {best_epoch}) | "
            f"{row['epoch_seconds']:.1f}s",
            flush=True,
        )

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break

    elapsed = time.time() - start_time
    summary = {
        "experiment": run_name,
        "model": config["model"]["name"],
        "pretrained": bool(config["model"].get("pretrained", False)),
        "augmentation": config["data"].get("augmentation", "none"),
        "image_size": int(config["data"]["image_size"]),
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "elapsed_seconds": elapsed,
        **split_info,
    }
    save_json(summary, run_dir / "run_summary.json")
    plot_history(history, run_dir / "curves.png")
    print(f"Completed {run_name}: best validation accuracy {best_val_accuracy:.4f}")
    print(f"Artifacts: {run_dir.resolve()}")


if __name__ == "__main__":
    main()

