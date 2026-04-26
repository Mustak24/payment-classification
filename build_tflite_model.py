"""Train a TensorFlow text classifier and export a .tflite model.

Usage:
    uv run build_tflite_model.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config
from data_loader import load_dataset, validate_class_balance
from preprocess import clean_batch


def _ensure_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "TensorFlow is required to build a .tflite model. "
            "Install it with: uv add tensorflow"
        ) from exc
    return tf


def _build_model(tf, vocab_size: int, sequence_length: int, num_classes: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="text"),
            tf.keras.layers.TextVectorization(
                max_tokens=vocab_size,
                output_mode="int",
                output_sequence_length=sequence_length,
                standardize=None,
                split="whitespace",
                name="vectorize",
            ),
            tf.keras.layers.Embedding(vocab_size, 64, name="embedding"),
            tf.keras.layers.GlobalAveragePooling1D(name="pool"),
            tf.keras.layers.Dropout(0.2, name="dropout"),
            tf.keras.layers.Dense(64, activation="relu", name="dense"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="probs"),
        ]
    )
    return model


def main() -> None:
    tf = _ensure_tensorflow()

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset()
    validate_class_balance(df)

    print("Cleaning text...")
    df["text_clean"] = clean_batch(df["text"].tolist())

    encoder = LabelEncoder()
    df["label_idx"] = encoder.fit_transform(df["label"])
    class_names = encoder.classes_.tolist()

    x_train, x_val, y_train, y_val = train_test_split(
        df["text_clean"].to_numpy(),
        df["label_idx"].to_numpy(),
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["label_idx"].to_numpy(),
    )

    vocab_size = 10_000
    sequence_length = 64
    batch_size = 32
    epochs = 10

    model = _build_model(
        tf=tf,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        num_classes=len(class_names),
    )

    vectorize_layer = model.get_layer("vectorize")
    vectorize_layer.adapt(x_train)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]

    print("Training TensorFlow model...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f} | loss: {val_loss:.4f}")

    keras_path = os.path.join(config.MODEL_DIR, "classifier_tflite.keras")
    tflite_path = os.path.join(config.MODEL_DIR, "classifier.tflite")
    labels_path = os.path.join(config.MODEL_DIR, "tflite_labels.json")

    model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": class_names,
                "confidence_threshold": config.CONFIDENCE_THRESHOLD,
                "trained_at": datetime.utcnow().isoformat() + "Z",
                "val_accuracy": round(float(val_acc), 4),
                "val_loss": round(float(val_loss), 4),
                "epochs_ran": len(history.history.get("loss", [])),
                "vocab_size": vocab_size,
                "sequence_length": sequence_length,
                "input_dtype": "string",
                "output": "softmax_probabilities",
                "requires_select_tf_ops": True,
            },
            f,
            indent=2,
        )

    print("Saved TensorFlow model:", keras_path)
    print("Saved TFLite model    :", tflite_path)
    print("Saved label metadata  :", labels_path)


if __name__ == "__main__":
    main()
