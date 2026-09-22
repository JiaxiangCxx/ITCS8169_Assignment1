from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torchvision import models


class TNet(nn.Module):
    """The deliberately weak grayscale CNN from the starter notebook."""

    def __init__(self, num_classes: int, image_size: int = 64) -> None:
        super().__init__()
        feature_size = (image_size - 2) // 4
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * feature_size * feature_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def _replace_classifier(model: nn.Module, num_classes: int) -> list[nn.Parameter]:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return list(model.fc.parameters())

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for index in range(len(model.classifier) - 1, -1, -1):
            layer = model.classifier[index]
            if isinstance(layer, nn.Linear):
                model.classifier[index] = nn.Linear(layer.in_features, num_classes)
                return list(model.classifier[index].parameters())

    raise ValueError(f"Unsupported classifier layout: {type(model).__name__}")


def build_model(
    model_config: dict,
    num_classes: int,
    image_size: int,
    load_pretrained: bool | None = None,
) -> tuple[nn.Module, list[nn.Parameter]]:
    name = str(model_config["name"])
    pretrained = bool(model_config.get("pretrained", False))
    if load_pretrained is not None:
        pretrained = load_pretrained

    if name == "tnet":
        model = TNet(num_classes=num_classes, image_size=image_size)
        return model, list(model.classifier.parameters())

    supported = {
        "resnet18",
        "resnet50",
        "efficientnet_b0",
        "mobilenet_v3_small",
        "convnext_tiny",
    }
    if name not in supported:
        raise ValueError(f"Unsupported model {name!r}; choose one of {sorted(supported | {'tnet'})}")

    weights = models.get_model_weights(name).DEFAULT if pretrained else None
    model = models.get_model(name, weights=weights)
    head_parameters = _replace_classifier(model, num_classes)
    return model, head_parameters


def set_backbone_trainable(
    model: nn.Module,
    head_parameters: list[nn.Parameter],
    trainable: bool,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in head_parameters:
        parameter.requires_grad = True


def clone_model_config_without_pretraining(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["pretrained"] = False
    return result


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable

