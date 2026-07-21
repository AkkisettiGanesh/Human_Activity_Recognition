"""
model.py
--------
Defines the LSTM-based deep learning architecture used to classify pose
landmark sequences into activity classes.
"""

from __future__ import annotations

from tensorflow.keras import layers, models, regularizers


def build_lstm_model(sequence_length: int, feature_dim: int, num_classes: int,
                      lstm_units: tuple[int, int] = (128, 64),
                      dropout_rate: float = 0.4,
                      l2_reg: float = 1e-4) -> models.Model:
    """Build and compile an LSTM classifier.

    Args:
        sequence_length: number of frames per input sequence.
        feature_dim: number of pose features per frame (33 landmarks x 4).
        num_classes: number of activity classes detected in the dataset.
        lstm_units: hidden units for the two stacked LSTM layers.
        dropout_rate: dropout applied after each LSTM/Dense block.
        l2_reg: L2 weight regularization strength.

    Returns:
        A compiled `tf.keras.Model`.
    """
    inputs = layers.Input(shape=(sequence_length, feature_dim), name="pose_sequence")

    x = layers.Masking(mask_value=0.0)(inputs)

    x = layers.LSTM(lstm_units[0], return_sequences=True,
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.LSTM(lstm_units[1], return_sequences=False,
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate / 2)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="activity")(x)

    model = models.Model(inputs, outputs, name="HAR_LSTM")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_lstm_model(sequence_length=30, feature_dim=132, num_classes=9)
    m.summary()
