# =============================================================================
# train.py — Full training pipeline (FastText)
# Run:  python train.py
# =============================================================================

import os
import json
import logging
import warnings
import sys
import threading
import time
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

import fasttext

import config
from data_loader import load_dataset, validate_class_balance
from preprocess import clean_batch

warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOG_DIR, "train.log")),
    ],
)
logger = logging.getLogger("train")


# ── Helpers ───────────────────────────────────────────────────────────────────

def write_fasttext_file(df: pd.DataFrame, output_path: str) -> None:
    """Write a DataFrame to FastText's labeled format.

    Each line: ``__label__<LABEL> <cleaned text>``
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            label = row["label"]
            text  = row["text_clean"]
            f.write(f"__label__{label} {text}\n")
    logger.info("Wrote FastText file → %s  (%d rows)", output_path, len(df))


def _patch_numpy_for_fasttext():
    """Monkey-patch numpy to work around fasttext-wheel's numpy 2.x incompatibility.

    fasttext internally calls ``np.array(probs, copy=False)`` which raises
    on numpy >= 2.0.  We patch ``np.array`` to fall back to ``np.asarray``
    when ``copy=False`` fails.
    """
    _original_np_array = np.array

    def _patched_array(*args, **kwargs):
        try:
            return _original_np_array(*args, **kwargs)
        except ValueError as exc:
            if "copy" in str(exc) and kwargs.get("copy") is False:
                kwargs.pop("copy")
                return np.asarray(*args, **kwargs)
            raise

    np.array = _patched_array


_patch_numpy_for_fasttext()


@contextmanager
def terminal_status(message: str, interval: float = 1.0):
    """Show a lightweight terminal heartbeat while a long task is running."""
    stop_event = threading.Event()
    started_at = time.monotonic()

    def _run():
        frames = "|/-\\"
        frame_index = 0
        while not stop_event.wait(interval):
            elapsed = int(time.monotonic() - started_at)
            sys.stdout.write(f"\r{message} {frames[frame_index % len(frames)]} {elapsed}s elapsed")
            sys.stdout.flush()
            frame_index += 1

    sys.stdout.write(f"{message}...\n")
    sys.stdout.flush()
    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    try:
        yield
    finally:
        stop_event.set()
        worker.join(timeout=2)
        sys.stdout.write(f"\r{message} done{' ' * 20}\n")
        sys.stdout.flush()


def predict_all(model, texts: list[str]) -> tuple[list[str], list[float]]:
    """Run FastText prediction on a list of texts, returning labels and confidences."""
    predicted_labels: list[str] = []
    confidences:      list[float] = []

    for text in texts:
        labels, probs = model.predict(text, k=1)
        predicted_labels.append(labels[0].replace("__label__", ""))
        confidences.append(float(probs[0]))

    return predicted_labels, confidences


def plot_confusion_matrix(y_true, y_pred, labels: list[str]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), max(5, len(labels) * 1.2)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(config.LOG_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved → %s", out_path)


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(max(6, len(counts) * 1.2), 4))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("Class")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(config.LOG_DIR, "class_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Class distribution plot saved → %s", out_path)


# ── Main training routine ─────────────────────────────────────────────────────

def train() -> None:
    logger.info("=" * 60)
    logger.info("SMS Classifier — FastText Training Pipeline")
    logger.info("=" * 60)

    # 1. Load & validate data
    df = load_dataset()
    validate_class_balance(df)
    plot_class_distribution(df)

    # 2. Preprocess text
    logger.info("Cleaning text...")
    df["text_clean"] = clean_batch(df["text"].tolist())

    # 3. Derive class names from the data
    class_names = sorted(df["label"].unique().tolist())
    logger.info("Classes: %s", class_names)

    # 4. Train / test split  (stratified, same as before)
    train_df, test_df = train_test_split(
        df[["text_clean", "label"]],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    logger.info("Train: %d | Test: %d", len(train_df), len(test_df))

    # 5. Write FastText-format training/test files
    write_fasttext_file(train_df, config.FASTTEXT_TRAIN_FILE)
    write_fasttext_file(test_df,  config.FASTTEXT_TEST_FILE)

    # 6. Train model
    if config.AUTOTUNE_DURATION > 0:
        logger.info(
            "Training with auto-tune (duration=%ds)...",
            config.AUTOTUNE_DURATION,
        )
        with terminal_status("FastText auto-tune running"):
            model = fasttext.train_supervised(
                input=config.FASTTEXT_TRAIN_FILE,
                autotuneValidationFile=config.FASTTEXT_TEST_FILE,
                autotuneDuration=config.AUTOTUNE_DURATION,
            )
    else:
        logger.info("Training with manual hyperparameters: %s", config.FASTTEXT_PARAMS)
        with terminal_status("FastText training running"):
            model = fasttext.train_supervised(
                input=config.FASTTEXT_TRAIN_FILE,
                **config.FASTTEXT_PARAMS,
            )

    # 7. Built-in test metrics
    n_test, precision_at1, recall_at1 = model.test(config.FASTTEXT_TEST_FILE)
    logger.info(
        "FastText test metrics — N: %d | P@1: %.4f | R@1: %.4f",
        n_test, precision_at1, recall_at1,
    )

    # 8. Detailed evaluation on held-out test set
    y_true = test_df["label"].tolist()
    y_pred, _ = predict_all(model, test_df["text_clean"].tolist())

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)

    logger.info("Test Accuracy : %.4f", acc)
    logger.info("\nClassification Report:\n%s", report_str)

    plot_confusion_matrix(y_true, y_pred, labels=class_names)

    # 9. Save full model (.bin)
    model.save_model(config.MODEL_PATH)
    logger.info("Saved full model → %s", config.MODEL_PATH)

    # 10. Quantize and save compressed model (.ftz)
    with terminal_status("Quantizing model"):
        model.quantize(input=config.FASTTEXT_TRAIN_FILE, retrain=True)
    model.save_model(config.MODEL_QUANTIZED_PATH)
    logger.info("Saved quantized model → %s", config.MODEL_QUANTIZED_PATH)

    # 11. Save metadata (human-readable)
    metadata = {
        "trained_at":            datetime.utcnow().isoformat() + "Z",
        "model_type":            "fasttext",
        "classes":               class_names,
        "n_train":               len(train_df),
        "n_test":                len(test_df),
        "autotune_enabled":      config.AUTOTUNE_DURATION > 0,
        "autotune_duration":     config.AUTOTUNE_DURATION,
        "fasttext_params":       config.FASTTEXT_PARAMS if config.AUTOTUNE_DURATION == 0 else "auto",
        "test_accuracy":         round(acc, 4),
        "test_precision_at1":    round(precision_at1, 4),
        "test_recall_at1":       round(recall_at1, 4),
        "test_f1_macro":         round(report["macro avg"]["f1-score"], 4),
        "confidence_threshold":  config.CONFIDENCE_THRESHOLD,
        "per_class_metrics": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1-score":  round(report[cls]["f1-score"], 4),
                "support":   report[cls]["support"],
            }
            for cls in class_names
        },
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved → %s", config.METADATA_PATH)

    logger.info("=" * 60)
    logger.info("Training complete. Test accuracy: %.2f%%", acc * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
