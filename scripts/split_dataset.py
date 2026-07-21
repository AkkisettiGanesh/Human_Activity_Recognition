"""
split_dataset.py
-----------------
Creates a stratified train / validation / test split over the detected
classes and video files, and writes the split to `processed_data/splits.json`
so that extract_landmarks.py / train.py can consume it consistently.

Usage:
    python scripts/split_dataset.py --dataset_dir dataset --train 0.7 --val 0.15 --test 0.15
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def build_split(dataset_dir: Path, train_ratio: float, val_ratio: float,
                 test_ratio: float, seed: int = 42) -> Dict[str, List[Dict]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    random.seed(seed)

    splits = {"train": [], "val": [], "test": []}
    class_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())

    for class_dir in class_dirs:
        videos = [str(p) for p in class_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
        random.shuffle(videos)

        n = len(videos)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for v in videos[:n_train]:
            splits["train"].append({"path": v, "label": class_dir.name})
        for v in videos[n_train:n_train + n_val]:
            splits["val"].append({"path": v, "label": class_dir.name})
        for v in videos[n_train + n_val:]:
            splits["test"].append({"path": v, "label": class_dir.name})

        logger.info("Class '%s': %d train / %d val / %d test", class_dir.name,
                     n_train, n_val, n - n_train - n_val)

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test split")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output", type=str, default="processed_data/splits.json")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = build_split(Path(args.dataset_dir), args.train, args.val, args.test, args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info("Total: %d train, %d val, %d test -> saved to %s",
                len(splits["train"]), len(splits["val"]), len(splits["test"]), args.output)


if __name__ == "__main__":
    main()
