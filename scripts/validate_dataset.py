"""
validate_dataset.py
--------------------
Validates every video in the dataset and quarantines corrupted /
unreadable files so they never reach the training pipeline.

Usage:
    python scripts/validate_dataset.py --dataset_dir dataset --quarantine_dir dataset_corrupted
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def is_valid_video(path: Path) -> bool:
    """Return True if OpenCV can open the file and decode at least one frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return ok


def validate_dataset(dataset_dir: Path, quarantine_dir: Path) -> None:
    dataset_dir = Path(dataset_dir)
    quarantine_dir = Path(quarantine_dir)

    total, corrupted = 0, 0
    for class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for video in class_dir.rglob("*"):
            if video.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            total += 1
            if not is_valid_video(video):
                corrupted += 1
                dest_dir = quarantine_dir / class_dir.name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(video), str(dest_dir / video.name))
                logger.warning("Quarantined corrupted video: %s", video)

    logger.info("Validation complete: %d/%d videos corrupted and removed", corrupted, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset videos")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--quarantine_dir", type=str, default="dataset_corrupted")
    args = parser.parse_args()
    validate_dataset(Path(args.dataset_dir), Path(args.quarantine_dir))


if __name__ == "__main__":
    main()
