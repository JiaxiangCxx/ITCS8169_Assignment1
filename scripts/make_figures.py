#!/usr/bin/env python3
"""Draw the report's learning-curve figure from saved run histories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Colour encodes the initialization and is the same in both panels.
IMAGENET = "#2a78d6"
RANDOM = "#eb6834"
PLACES365 = "#1baf7a"
REFERENCE = "#898781"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("report/figures/curves.pdf"))
    return parser.parse_args()


def read_history(path: Path) -> dict[str, list[float]]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def style_axis(axis, title: str, ylim: tuple[float, float]) -> None:
    axis.set_title(title, loc="left", fontsize=8.5, color=INK, pad=4)
    axis.set_xlabel("Epoch", fontsize=8, color=INK_SECONDARY, labelpad=1)
    axis.set_ylabel("Accuracy (%)", fontsize=8, color=INK_SECONDARY, labelpad=2)
    axis.set_ylim(*ylim)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
        axis.spines[side].set_linewidth(0.6)
    axis.tick_params(colors=INK_SECONDARY, labelsize=7.5, length=2, width=0.6)


def plot_run(
    axis,
    history: dict[str, list[float]],
    color: str,
    label: str,
    label_xy: tuple[float, float],
    show_train: bool,
) -> None:
    epochs = history["epoch"]
    validation = [100 * value for value in history["val_accuracy"]]
    axis.plot(epochs, validation, color=color, linewidth=1.5, solid_capstyle="round")
    if show_train:
        axis.plot(
            epochs,
            [100 * value for value in history["train_accuracy"]],
            color=color,
            linewidth=1.0,
            linestyle=(0, (3, 2)),
        )
    best = max(validation)
    best_epoch = epochs[validation.index(best)]
    axis.plot(
        [best_epoch],
        [best],
        marker="o",
        markersize=4.5,
        color=color,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=3,
    )
    axis.annotate(
        f"{label}: {best:.1f}%",
        xy=label_xy,
        fontsize=7,
        color=INK_SECONDARY,
        va="center",
    )


def main() -> None:
    args = parse_args()
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    runs = {
        name: read_history(args.runs_root / name / "history.csv")
        for name in (
            "baseline_tnet_run3",
            "resnet18_scratch",
            "resnet18_finetune_basic_run3",
            "resnet50_imagenet_finetune",
            "places365_resnet50_finetune",
        )
    }

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 2.15), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    style_axis(left, "(a) Initialization, ResNet-18 vs. starter TNet", (0, 104))
    plot_run(left, runs["resnet18_finetune_basic_run3"], IMAGENET,
             "ResNet-18, ImageNet", (25, 88), show_train=True)
    plot_run(left, runs["resnet18_scratch"], RANDOM,
             "ResNet-18, random", (50, 66), show_train=True)
    plot_run(left, runs["baseline_tnet_run3"], REFERENCE,
             "TNet (starter)", (22, 40), show_train=True)
    left.legend(
        handles=[
            Line2D([], [], color=INK_SECONDARY, linewidth=1.5, label="validation"),
            Line2D([], [], color=INK_SECONDARY, linewidth=1.0,
                   linestyle=(0, (3, 2)), label="training"),
            Line2D([], [], color=IMAGENET, linewidth=1.5, label="ImageNet init"),
            Line2D([], [], color=RANDOM, linewidth=1.5, label="random init"),
            Line2D([], [], color=REFERENCE, linewidth=1.5, label="starter TNet"),
        ],
        loc="lower right",
        fontsize=6.5,
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.0,
    )

    style_axis(right, "(b) Pretraining source, ResNet-50, same recipe", (60, 101))
    imagenet = runs["resnet50_imagenet_finetune"]
    places = runs["places365_resnet50_finetune"]
    plot_run(right, places, PLACES365, "Places365",
             (places["epoch"][-1] + 0.6, 100 * places["val_accuracy"][-1] + 1.2),
             show_train=False)
    plot_run(right, imagenet, IMAGENET, "ImageNet",
             (imagenet["epoch"][-1] + 0.6, 100 * imagenet["val_accuracy"][-1] - 1.2),
             show_train=False)
    right.set_xlim(0, max(places["epoch"][-1], imagenet["epoch"][-1]) + 9)
    right.legend(
        handles=[
            Line2D([], [], color=PLACES365, linewidth=1.5, label="Places365 init"),
            Line2D([], [], color=IMAGENET, linewidth=1.5, label="ImageNet init"),
        ],
        loc="lower right",
        fontsize=6.5,
        frameon=False,
    )

    figure.tight_layout(pad=0.3, w_pad=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    figure.savefig(args.output.with_suffix(".png"), dpi=200)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
