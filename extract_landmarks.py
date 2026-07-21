"""
extract_landmarks.py
--------------------
Utility functions to build the MediaPipe Pose Landmarker and extract pose landmarks
from video files or frame buffers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Total dimensions: 33 keypoints * 4 values (x, y, z, visibility) = 132
FRAME_FEATURE_DIM = 132

logger = logging.getLogger(__name__)


def build_landmarker() -> Tuple[mp.tasks.vision.PoseLandmarker, type[mp]]:
    """Build and return a MediaPipe PoseLandmarker instance along with the mp module."""
    model_dir = Path("models")
    
    # Auto-detect any available .task file in the models directory
    task_files = list(model_dir.glob("*.task"))
    
    if not task_files:
        raise FileNotFoundError(
            f"No pose landmarker model (.task file) found in {model_dir.resolve()}. "
            "Please ensure pose_landmarker.task or pose_landmarker_lite.task is in the models/ directory."
        )

    model_path = task_files[0]

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    return landmarker, mp


def extract_landmarks_from_video(
    video_path: str,
    landmarker: Optional[mp.tasks.vision.PoseLandmarker] = None,
    mp_module: Optional[type[mp]] = None,
    target_length: int = 30,
) -> Optional[np.ndarray]:
    """Extract pose landmark sequences from a video file using a fresh MediaPipe instance."""
    fresh_landmarker, mp_mod = build_landmarker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Could not open video file: %s", video_path)
        fresh_landmarker.close()
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    frame_time_ms = 1000.0 / fps
    landmarks_list = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb)
        
        timestamp_ms = int(frame_idx * frame_time_ms)
        frame_idx += 1

        result = fresh_landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            coords = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
            
            hip_center = (coords[23, :3] + coords[24, :3]) / 2.0
            coords[:, :3] -= hip_center
            landmarks_list.append(coords.flatten())
        else:
            landmarks_list.append(np.zeros(FRAME_FEATURE_DIM, dtype=np.float32))

    cap.release()
    fresh_landmarker.close()

    if not landmarks_list:
        return None

    raw_sequence = np.stack(landmarks_list)
    return resample_sequence(raw_sequence, target_length)


def resample_sequence(sequence: np.ndarray, target_length: int = 30) -> np.ndarray:
    """Resample a 2D sequence array along axis 0 to match target_length using linear interpolation."""
    current_length = sequence.shape[0]
    if current_length == target_length:
        return sequence.astype(np.float32)

    if current_length == 1:
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    original_indices = np.linspace(0, current_length - 1, num=current_length)
    target_indices = np.linspace(0, current_length - 1, num=target_length)

    resampled = np.zeros((target_length, sequence.shape[1]), dtype=np.float32)
    for col in range(sequence.shape[1]):
        resampled[:, col] = np.interp(target_indices, original_indices, sequence[:, col])

    return resampled