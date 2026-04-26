"""Retrain model by mining uncertain/incorrect rows from test CSV logs.

Workflow:
1. Read a test prediction CSV (default: test/log/model_test_predictions.csv)
2. Select rows where prediction is uncertain OR incorrect
3. Append selected message text to class files based on ground-truth `label`
4. Trigger the existing training pipeline (`train.train()`)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import config
from train import train


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = ROOT_DIR / "test" / "log" / "model_test_predictions.csv"


@dataclass
class RowSample:
    label: str
    text: str
    is_uncertain: bool
    correct: bool


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _resolve_data_file(filename: str) -> Path:
    candidates = [Path(config.DATA_DIR) / filename]

    if filename.lower() == "unknown.txt":
        candidates.append(Path(config.DATA_DIR) / "unknow.txt")
    elif filename.lower() == "unknow.txt":
        candidates.append(Path(config.DATA_DIR) / "unknown.txt")

    for path in candidates:
        if path.exists():
            return path

    # If neither spelling exists, create the primary configured file path.
    return candidates[0]


def _read_candidate_rows(csv_path: Path) -> list[RowSample]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV log not found: {csv_path}")

    selected: list[RowSample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "text", "is_uncertain", "correct"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row in reader:
            label = str(row.get("label", "")).strip()
            text = str(row.get("text", "")).strip()
            if not label or not text:
                continue

            is_uncertain = _to_bool(row.get("is_uncertain", False))
            correct = _to_bool(row.get("correct", True))

            if is_uncertain or not correct:
                selected.append(
                    RowSample(
                        label=label,
                        text=text,
                        is_uncertain=is_uncertain,
                        correct=correct,
                    )
                )

    return selected


def _load_existing_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def append_samples_to_training_data(samples: list[RowSample]) -> dict[str, int]:
    append_counts: dict[str, int] = {}

    grouped: dict[str, list[str]] = {}
    for sample in samples:
        grouped.setdefault(sample.label, []).append(sample.text)

    for label, texts in grouped.items():
        if label not in config.DATA_FILES:
            raise ValueError(
                f"Label '{label}' is not configured in config.DATA_FILES. "
                "Update config.DATA_FILES or clean the CSV labels."
            )

        data_file = _resolve_data_file(config.DATA_FILES[label])
        data_file.parent.mkdir(parents=True, exist_ok=True)

        existing = _load_existing_lines(data_file)
        unique_new: list[str] = []
        seen_in_batch: set[str] = set()

        for text in texts:
            if text in existing or text in seen_in_batch:
                continue
            unique_new.append(text)
            seen_in_batch.add(text)

        if unique_new:
            with data_file.open("a", encoding="utf-8") as handle:
                for text in unique_new:
                    handle.write(f"{text}\n")

        append_counts[label] = len(unique_new)

    return append_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read model test CSV logs, add uncertain/incorrect rows to training data, "
            "and retrain the model."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to model_test_predictions.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be appended, but do not update data or retrain.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Update training data but skip calling train().",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = _read_candidate_rows(args.csv_path)
    total_candidates = len(samples)

    uncertain_count = sum(1 for s in samples if s.is_uncertain)
    incorrect_count = sum(1 for s in samples if not s.correct)

    print(f"CSV path         : {args.csv_path}")
    print(f"Selected rows    : {total_candidates}")
    print(f"Uncertain rows   : {uncertain_count}")
    print(f"Incorrect rows   : {incorrect_count}")

    if total_candidates == 0:
        print("No uncertain/incorrect rows found. Nothing to append.")
        return

    if args.dry_run:
        print("Dry run enabled. No files updated, no training executed.")
        return

    append_counts = append_samples_to_training_data(samples)
    total_appended = sum(append_counts.values())

    print("Appended rows by label:")
    for label in sorted(append_counts):
        print(f"  {label}: {append_counts[label]}")
    print(f"Total newly appended: {total_appended}")

    if total_appended == 0:
        print("All selected rows already exist in training data. Skipping retrain.")
        return

    if args.skip_train:
        print("--skip-train enabled. Data updated; retraining skipped.")
        return

    print("Starting retraining...")
    train()


if __name__ == "__main__":
    main()
