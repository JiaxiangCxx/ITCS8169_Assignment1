from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .utils import save_json, seed_worker


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_transforms(data_config: dict, training: bool) -> transforms.Compose:
    image_size = int(data_config["image_size"])
    color_mode = str(data_config.get("color_mode", "rgb"))
    augmentation = str(data_config.get("augmentation", "none"))
    interpolation_name = str(data_config.get("interpolation", "bilinear")).upper()
    try:
        interpolation = transforms.InterpolationMode[interpolation_name]
    except KeyError as error:
        raise ValueError(f"Unknown interpolation mode: {interpolation_name}") from error
    operations: list = []

    if color_mode == "grayscale":
        operations.append(transforms.Grayscale(num_output_channels=1))
        if training and augmentation != "none":
            operations.extend(
                [
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.75, 1.0), interpolation=interpolation
                    ),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            operations.append(
                transforms.Resize((image_size, image_size), interpolation=interpolation)
            )
        operations.extend(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )
        return transforms.Compose(operations)

    if color_mode != "rgb":
        raise ValueError("data.color_mode must be 'rgb' or 'grayscale'")

    if training and augmentation == "light":
        operations.extend(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.85, 1.0), interpolation=interpolation
                ),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif training and augmentation == "basic":
        operations.extend(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.75, 1.0), interpolation=interpolation
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    elif training and augmentation == "strong":
        operations.extend(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.60, 1.0), interpolation=interpolation
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.35,
                            contrast=0.35,
                            saturation=0.35,
                            hue=0.08,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.05),
                transforms.RandomRotation(degrees=10),
            ]
        )
    elif training and augmentation != "none":
        raise ValueError(f"Unknown augmentation profile: {augmentation}")
    else:
        resize_size = int(
            data_config.get("resize_size", math.ceil(image_size / 0.875))
        )
        operations.extend(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(image_size),
            ]
        )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def _dataset_fingerprint(dataset: datasets.ImageFolder) -> str:
    paths = [str(Path(path).relative_to(dataset.root)) for path, _ in dataset.samples]
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def _stratified_split(
    targets: list[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for target in sorted(by_class):
        indices = by_class[target]
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_fraction)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def load_or_create_split(
    dataset: datasets.ImageFolder,
    split_file: str | Path,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    split_file = Path(split_file)
    fingerprint = _dataset_fingerprint(dataset)
    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved["dataset_fingerprint"] != fingerprint:
            raise RuntimeError(
                f"Dataset contents changed after {split_file} was created; "
                "use a new split filename or inspect the data"
            )
        if int(saved["seed"]) != seed or float(saved["val_fraction"]) != val_fraction:
            raise RuntimeError(f"Split settings do not match existing file: {split_file}")
        return list(saved["train_indices"]), list(saved["val_indices"])

    train_indices, val_indices = _stratified_split(
        list(dataset.targets), val_fraction=val_fraction, seed=seed
    )
    payload = {
        "seed": seed,
        "val_fraction": val_fraction,
        "dataset_fingerprint": fingerprint,
        "dataset_size": len(dataset),
        "class_names": dataset.classes,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    save_json(payload, split_file)
    return train_indices, val_indices


def make_train_val_loaders(
    data_config: dict,
    seed: int,
) -> tuple[DataLoader, DataLoader, list[str], dict]:
    data_root = Path(data_config["root"])
    train_root = data_root / str(data_config.get("train_dir", "train"))
    if not train_root.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_root}")

    training_dataset = datasets.ImageFolder(
        train_root, transform=build_transforms(data_config, training=True)
    )
    validation_dataset = datasets.ImageFolder(
        train_root, transform=build_transforms(data_config, training=False)
    )
    if training_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise RuntimeError("Training and validation class mappings differ")

    train_indices, val_indices = load_or_create_split(
        validation_dataset,
        split_file=data_config["split_file"],
        val_fraction=float(data_config["val_fraction"]),
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_arguments = {
        "batch_size": int(data_config["batch_size"]),
        "num_workers": int(data_config.get("num_workers", 4)),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": int(data_config.get("num_workers", 4)) > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        Subset(training_dataset, train_indices),
        shuffle=True,
        generator=generator,
        **loader_arguments,
    )
    val_loader = DataLoader(
        Subset(validation_dataset, val_indices),
        shuffle=False,
        **loader_arguments,
    )
    split_info = {
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "split_file": str(data_config["split_file"]),
    }
    return train_loader, val_loader, training_dataset.classes, split_info


def make_labeled_loader(
    data_config: dict,
    split: str,
) -> tuple[DataLoader, list[str]]:
    data_root = Path(data_config["root"])
    split_root = data_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Labeled split directory not found: {split_root}")
    dataset = datasets.ImageFolder(
        split_root, transform=build_transforms(data_config, training=False)
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(data_config.get("num_workers", 4)) > 0,
    )
    return loader, dataset.classes


class FlatImageDataset(Dataset):
    def __init__(self, root: str | Path, transform) -> None:
        self.root = Path(root)
        self.paths = sorted(
            path
            for path in self.root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {self.root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, path.name
