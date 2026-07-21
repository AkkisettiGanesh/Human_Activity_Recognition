"""
predict.py
----------
Loads trained LSTM model and runs activity prediction on uploaded videos or live webcam frames.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

from extract_landmarks import (
    FRAME_FEATURE_DIM,
    build_landmarker,
    extract_landmarks_from_video,
    resample_sequence,
)


class ActivityPredictor:
    def __init__(self, model_dir: str = "models", data_dir: str = "processed_data"):
        import tensorflow as tf

        model_dir = Path(model_dir)
        data_dir = Path(data_dir)

        self.model = tf.keras.models.load_model(model_dir / "lstm_model.keras")

        with open(model_dir / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(data_dir / "metadata.json") as f:
            metadata = json.load(f)
        self.sequence_length = metadata["sequence_length"]
        self.class_names = metadata["classes"]

        self.landmarker, self.mp = build_landmarker()
        self._live_buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self._timestamp_ms = 0

    def predict_video(self, video_path: str) -> Tuple[str, float, np.ndarray]:
        sequence = extract_landmarks_from_video(
            video_path, self.landmarker, self.mp, self.sequence_length
        )
        if sequence is None:
            raise ValueError(f"Could not extract pose landmarks from {video_path}")

        probs = self.model.predict(sequence[np.newaxis, ...], verbose=0)[0]
        label_idx = int(np.argmax(probs))
        label = self.label_encoder.inverse_transform([label_idx])[0]
        return label, float(probs[label_idx]), probs

    def push_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

        current_time_ms = int(time.time() * 1000)
        if current_time_ms <= self._timestamp_ms:
            current_time_ms = self._timestamp_ms + 1

        self._timestamp_ms = current_time_ms
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            coords = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
            hip_center = (coords[23, :3] + coords[24, :3]) / 2.0
            coords[:, :3] -= hip_center
            self._live_buffer.append(coords.flatten())
            return coords
        else:
            self._live_buffer.append(np.zeros(FRAME_FEATURE_DIM, dtype=np.float32))
            return None

    def predict_live(self) -> Optional[Tuple[str, float, np.ndarray]]:
        """Predict activity from current live buffer, returning label, confidence, and all probabilities."""
        if len(self._live_buffer) < self.sequence_length:
            return None
        sequence = resample_sequence(np.stack(self._live_buffer), self.sequence_length)
        probs = self.model.predict(sequence[np.newaxis, ...], verbose=0)[0]
        label_idx = int(np.argmax(probs))
        label = self.label_encoder.inverse_transform([label_idx])[0]
        return label, float(probs[label_idx]), probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict activity for a video file")
    parser.add_argument("video_path", type=str)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--data_dir", type=str, default="processed_data")
    args = parser.parse_args()

    predictor = ActivityPredictor(args.model_dir, args.data_dir)
    label, confidence, probs = predictor.predict_video(args.video_path)

    print(f"Predicted activity: {label} ({confidence * 100:.2f}% confidence)")
    for name, p in sorted(zip(predictor.class_names, probs), key=lambda x: -x[1]):
        print(f"  {name:15s} {p * 100:5.2f}%")


if __name__ == "__main__":
    main()