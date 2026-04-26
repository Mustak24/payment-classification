"""Run an end-to-end check of the saved SMS classifier.

The script loads labeled text files for CREDIT, DEBIT, and UNKNOWN,
runs batch predictions with the trained model, and writes results into
test/log so the output can be reviewed or shared.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from predict import get_classifier


DEFAULT_INPUT_DIR = ROOT_DIR / "test" / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "test" / "log"


def resolve_data_file(input_dir: Path, filename: str) -> Path:
    """Return the first matching file for a configured class file name."""
    candidates = [input_dir / filename]

    if filename.lower() == "unknown.txt":
        candidates.append(input_dir / "unknow.txt")
    elif filename.lower() == "unknow.txt":
        candidates.append(input_dir / "unknown.txt")

    fallback_dir = Path(config.DATA_DIR)
    for candidate in list(candidates):
        if candidate.exists():
            return candidate
        fallback = fallback_dir / candidate.name
        if fallback.exists():
            return fallback

    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing test data file. Expected one of: {expected}")


def load_labeled_samples(input_dir: Path) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []

    for label, filename in config.DATA_FILES.items():
        path = resolve_data_file(input_dir, filename)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                samples.append({"text": text, "label": label, "source_file": path.name})

    if not samples:
        raise ValueError("No test samples found in the configured input directory.")

    return samples


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "source_file",
        "label",
        "text",
        "predicted_class",
        "confidence",
        "is_uncertain",
        "correct",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(
    output_dir: Path,
    true_labels: list[str],
    predicted_labels: list[str],
    class_labels: list[str],
) -> Path:
    labels = list(class_labels)
    if "UNCERTAIN" in predicted_labels and "UNCERTAIN" not in labels:
        labels.append("UNCERTAIN")

    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    fig_width = max(6, len(labels) * 1.4)
    fig_height = max(5, len(labels) * 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix — Model Test", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "model_test_confusion_matrix.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    return image_path


def evaluate(input_dir: Path, output_dir: Path) -> dict[str, object]:
    classifier = get_classifier()
    samples = load_labeled_samples(input_dir)
    predictions = classifier.predict_batch([sample["text"] for sample in samples])

    rows: list[dict[str, object]] = []
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    total = len(samples)
    correct = 0
    uncertain = 0
    per_label_totals: dict[str, int] = {}
    per_label_correct: dict[str, int] = {}

    for sample, prediction in zip(samples, predictions):
        is_correct = prediction.predicted_class == sample["label"]
        true_labels.append(sample["label"])
        predicted_labels.append(prediction.predicted_class)
        correct += int(is_correct)
        uncertain += int(prediction.is_uncertain)
        per_label_totals[sample["label"]] = per_label_totals.get(sample["label"], 0) + 1
        per_label_correct[sample["label"]] = per_label_correct.get(sample["label"], 0) + int(is_correct)

        rows.append(
            {
                "source_file": sample["source_file"],
                "label": sample["label"],
                "text": sample["text"],
                "predicted_class": prediction.predicted_class,
                "confidence": prediction.confidence,
                "is_uncertain": prediction.is_uncertain,
                "correct": is_correct,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_test_predictions.csv"
    json_path = output_dir / "model_test_summary.json"
    confusion_matrix_path = save_confusion_matrix(
        output_dir=output_dir,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_labels=list(classifier.metadata.get("classes", [])) or sorted(per_label_totals),
    )

    write_csv(csv_path, rows)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "uncertain_predictions": uncertain,
        "uncertain_rate": round(uncertain / total, 4) if total else 0.0,
        "per_label_accuracy": {
            label: round(per_label_correct[label] / per_label_totals[label], 4)
            for label in sorted(per_label_totals)
        },
        "per_label_totals": dict(sorted(per_label_totals.items())),
        "confusion_matrix_path": str(confusion_matrix_path),
        "classes": list(classifier.metadata.get("classes", [])),
        "metadata": classifier.metadata,
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Model test complete")
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Accuracy  : {summary['accuracy']:.2%}")
    print(f"Uncertain : {summary['uncertain_predictions']} / {summary['total_samples']}")
    print(f"CSV log   : {csv_path}")
    print(f"JSON log  : {json_path}")
    print(f"Matrix    : {confusion_matrix_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the saved SMS classifier on labeled files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing credit.txt, debit.txt, and unknow/unknown.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the test results are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()