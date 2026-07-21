"""
train.py
--------
Trains the LSTM activity-recognition model on the pose landmark sequences
produced by extract_landmarks.py, and generates full evaluation artifacts
(accuracy/loss curves, confusion matrix, classification report).

Usage:
    python train.py --data_dir processed_data --epochs 100 --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import build_lstm_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_split(data_dir: Path, split: str):
    X = np.load(data_dir / f"X_{split}.npy")
    y = np.load(data_dir / f"y_{split}.npy")
    return X, y


def plot_history(history, output_dir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(history.history["accuracy"], label="Train Accuracy")
    ax[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax[0].set_title("Model Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    ax[1].plot(history.history["loss"], label="Train Loss")
    ax[1].plot(history.history["val_loss"], label="Validation Loss")
    ax[1].set_title("Model Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=150)
    fig.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HAR LSTM model")
    parser.add_argument("--data_dir", type=str, default="processed_data")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(data_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    class_names = metadata["classes"]
    sequence_length = metadata["sequence_length"]
    feature_dim = metadata["feature_dim"]

    X_train, y_train = load_split(data_dir, "train")
    X_val, y_val = load_split(data_dir, "val")
    X_test, y_test = load_split(data_dir, "test")

    logger.info("Train: %s, Val: %s, Test: %s", X_train.shape, X_val.shape, X_test.shape)

    model = build_lstm_model(sequence_length, feature_dim, num_classes=len(class_names))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
        ModelCheckpoint(str(model_dir / "lstm_model.keras"), monitor="val_accuracy",
                         save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    plot_history(history, output_dir)

    # Final evaluation on the held-out test set
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    logger.info("\n%s", report)
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)

    plot_confusion_matrix(y_test, y_pred, class_names, output_dir)

    with open(model_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    training_summary = {
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "test_accuracy": float(np.mean(y_pred == y_test)),
        "epochs_trained": len(history.history["loss"]),
        "num_classes": len(class_names),
        "class_names": class_names,
        "sequence_length": sequence_length,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)

    logger.info("Training complete. Test accuracy: %.4f", training_summary["test_accuracy"])


if __name__ == "__main__":
    main()
