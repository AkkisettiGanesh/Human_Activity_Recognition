"""
inspect_dataset.py
-------------------
Dynamically inspects the extracted dataset folder and produces statistics.

The dataset is expected to sit under `dataset/` with one sub-folder per
activity class (the class name is auto-detected from the folder name; no
activity names are hard-coded anywhere in this project).

Usage:
    python scripts/inspect_dataset.py --dataset_dir dataset --output_dir outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def find_video_files(class_dir: Path) -> List[Path]:
    """Recursively find every video file under a class directory (handles
    nested folders such as `clap/clap/*.avi`)."""
    return [p for p in class_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]


def probe_video(path: Path) -> Dict:
    """Open a video with OpenCV and pull basic metadata. Returns a dict with
    `readable=False` if the file cannot be decoded (i.e. corrupted)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return {"readable": False}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # A frame_count of 0 usually means the container index is broken even
    # though the file opened -- try to actually read one frame to confirm.
    ok, _ = cap.read()
    cap.release()

    if not ok:
        return {"readable": False}

    duration = frame_count / fps if fps and fps > 0 else 0.0
    return {
        "readable": True,
        "frame_count": frame_count,
        "fps": round(fps, 2) if fps else 0.0,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 2),
    }


def inspect_dataset(dataset_dir: Path) -> Dict:
    """Detect every class folder + every video inside it and build a full
    statistics dictionary. Adapts automatically to whatever classes exist."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class sub-folders found inside {dataset_dir}")

    class_names = [d.name for d in class_dirs]
    logger.info("Detected %d classes: %s", len(class_names), class_names)

    stats = {
        "dataset_dir": str(dataset_dir),
        "num_classes": len(class_names),
        "class_names": class_names,
        "per_class": {},
        "corrupted_files": [],
        "total_videos": 0,
        "total_valid_videos": 0,
    }

    for class_dir in class_dirs:
        videos = find_video_files(class_dir)
        readable, corrupted = [], []
        durations, fps_list, resolutions = [], [], []

        for v in videos:
            meta = probe_video(v)
            if meta["readable"]:
                readable.append(v)
                durations.append(meta["duration_sec"])
                fps_list.append(meta["fps"])
                resolutions.append((meta["width"], meta["height"]))
            else:
                corrupted.append(str(v))
                stats["corrupted_files"].append(str(v))

        stats["per_class"][class_dir.name] = {
            "total_videos": len(videos),
            "valid_videos": len(readable),
            "corrupted_videos": len(corrupted),
            "avg_duration_sec": round(sum(durations) / len(durations), 2) if durations else 0,
            "avg_fps": round(sum(fps_list) / len(fps_list), 2) if fps_list else 0,
            "common_resolution": max(set(resolutions), key=resolutions.count) if resolutions else None,
        }
        stats["total_videos"] += len(videos)
        stats["total_valid_videos"] += len(readable)

        logger.info(
            "Class '%s': %d videos (%d valid, %d corrupted)",
            class_dir.name, len(videos), len(readable), len(corrupted),
        )

    return stats


def plot_class_distribution(stats: Dict, output_path: Path) -> None:
    classes = stats["class_names"]
    counts = [stats["per_class"][c]["valid_videos"] for c in classes]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color="#4C9AFF")
    plt.title("Dataset Statistics - Valid Videos per Class")
    plt.xlabel("Activity Class")
    plt.ylabel("Number of Videos")
    plt.xticks(rotation=45, ha="right")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                  str(count), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved class distribution chart -> %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the HAR dataset")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = inspect_dataset(dataset_dir)

    with open(output_dir / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    class_mapping = {name: idx for idx, name in enumerate(stats["class_names"])}
    with open("processed_data/class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)

    plot_class_distribution(stats, output_dir / "dataset_statistics.png")

    report_lines = [
        "DATASET PREPROCESSING REPORT",
        "=" * 40,
        f"Dataset directory : {stats['dataset_dir']}",
        f"Total classes      : {stats['num_classes']}",
        f"Total videos found : {stats['total_videos']}",
        f"Valid videos       : {stats['total_valid_videos']}",
        f"Corrupted videos   : {len(stats['corrupted_files'])}",
        "",
        "Per-class breakdown:",
    ]
    for c in stats["class_names"]:
        pc = stats["per_class"][c]
        report_lines.append(
            f"  - {c:15s} total={pc['total_videos']:4d}  valid={pc['valid_videos']:4d}  "
            f"corrupted={pc['corrupted_videos']:3d}  avg_dur={pc['avg_duration_sec']}s  "
            f"avg_fps={pc['avg_fps']}  res={pc['common_resolution']}"
        )
    report = "\n".join(report_lines)
    with open(output_dir / "dataset_preprocessing_report.txt", "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
