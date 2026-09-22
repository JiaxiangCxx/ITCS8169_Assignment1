#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine experiment run summaries")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output", default="runs/experiment_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for path in sorted(Path(args.runs_root).glob("*/run_summary.json")):
        with path.open("r", encoding="utf-8") as stream:
            row = json.load(stream)
        row["summary_path"] = str(path)
        summaries.append(row)
    if not summaries:
        raise SystemExit(f"No run_summary.json files found below {args.runs_root}")

    preferred = [
        "experiment",
        "model",
        "pretrained",
        "augmentation",
        "image_size",
        "seed",
        "epochs_completed",
        "best_epoch",
        "best_val_accuracy",
        "elapsed_seconds",
        "train_size",
        "val_size",
        "summary_path",
    ]
    fieldnames = preferred + sorted(set().union(*summaries) - set(preferred))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {len(summaries)} summaries to {output.resolve()}")


if __name__ == "__main__":
    main()

