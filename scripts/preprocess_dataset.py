"""
preprocess_dataset.py
----------------------
Runs the complete preprocessing pipeline end-to-end:
    1. inspect_dataset.py  -> dataset statistics + report
    2. validate_dataset.py -> quarantine corrupted videos
    3. split_dataset.py    -> stratified train/val/test split
    4. extract_landmarks.py (project root) -> X.npy / y.npy feature sequences

Usage:
    python scripts/preprocess_dataset.py --dataset_dir dataset
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_step(description: str, cmd: list[str]) -> None:
    logger.info("=== %s ===", description)
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {description}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full preprocessing pipeline")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--sequence_length", type=int, default=30)
    args = parser.parse_args()

    py = sys.executable

    run_step("1/4 Inspecting dataset", [py, "scripts/inspect_dataset.py", "--dataset_dir", args.dataset_dir])
    run_step("2/4 Validating videos", [py, "scripts/validate_dataset.py", "--dataset_dir", args.dataset_dir])
    run_step("3/4 Splitting dataset", [py, "scripts/split_dataset.py", "--dataset_dir", args.dataset_dir])
    run_step("4/4 Extracting MediaPipe pose landmarks", [
        py, "extract_landmarks.py",
        "--splits", "processed_data/splits.json",
        "--sequence_length", str(args.sequence_length),
    ])

    logger.info("Preprocessing pipeline complete. Run `python train.py` next.")


if __name__ == "__main__":
    main()
