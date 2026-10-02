#!/usr/bin/env python3
"""Write a weights-only copy of a training checkpoint for distribution.

The copy keeps everything evaluate.py, predict.py and ensemble.py read
(weights, resolved configuration, class order, epoch and validation accuracy)
and drops the optimizer and scheduler state, which only matter for resuming
training.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


KEEP = (
    "epoch",
    "best_val_accuracy",
    "model_state",
    "config",
    "class_names",
    "split_info",
    "torch_version",
    "torchvision_version",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip a checkpoint to its weights")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    slim = {key: checkpoint[key] for key in KEEP if key in checkpoint}
    slim["source_checkpoint"] = str(Path(args.checkpoint))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"{output}  {output.stat().st_size / 2**20:.1f} MiB  sha256 {digest}")


if __name__ == "__main__":
    main()
