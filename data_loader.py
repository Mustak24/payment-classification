"""Utilities for loading the labeled SMS training dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def _resolve_data_file(filename: str) -> Path:
    """Return the first existing path for a configured data filename."""
    candidates = [Path(config.DATA_DIR) / filename]

    if filename.lower() == "unknown.txt":
        candidates.append(Path(config.DATA_DIR) / "unknow.txt")
    elif filename.lower() == "unknow.txt":
        candidates.append(Path(config.DATA_DIR) / "unknown.txt")

    for path in candidates:
        if path.exists():
            return path

    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing dataset file. Expected one of: {expected}")


def _read_messages(path: Path, label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append({"text": text, "label": label})

    return rows


def load_dataset() -> pd.DataFrame:
    """Load all class files into a single DataFrame with `text` and `label`."""
    rows: list[dict[str, str]] = []

    for label, filename in config.DATA_FILES.items():
        path = _resolve_data_file(filename)
        rows.extend(_read_messages(path, label))

    if not rows:
        raise ValueError("No training data found in configured data files.")

    df = pd.DataFrame(rows, columns=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df


def validate_class_balance(df: pd.DataFrame) -> None:
    """Validate that each class has enough samples for the training pipeline."""
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    counts = df["label"].value_counts().sort_index()
    if counts.empty:
        raise ValueError("Dataset is empty.")

    print("Class balance:")
    for label, count in counts.items():
        print(f"  {label}: {count}")

    minimum_required = max(config.CV_FOLDS + 1, 6)
    too_small = counts[counts < minimum_required]
    if not too_small.empty:
        details = ", ".join(f"{label}={count}" for label, count in too_small.items())
        raise ValueError(
            "Each class needs at least "
            f"{minimum_required} samples for train/test split and {config.CV_FOLDS}-fold CV. "
            f"Found: {details}"
        )
