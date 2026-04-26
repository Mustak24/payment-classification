# =============================================================================
# train.py — Full training pipeline
# Run:  python train.py
# =============================================================================

import os
import json
import logging
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

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

def save_artifact(obj, path: str, name: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved %s → %s", name, path)


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
    logger.info("SMS Classifier — Training Pipeline")
    logger.info("=" * 60)

    # 1. Load & validate data
    df = load_dataset()
    validate_class_balance(df)
    plot_class_distribution(df)

    # 2. Preprocess text
    logger.info("Cleaning text...")
    df["text_clean"] = clean_batch(df["text"].tolist())

    # 3. Encode labels
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    class_names = list(le.classes_)
    logger.info("Classes: %s", class_names)

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"],
        df["label_enc"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["label_enc"],
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # 5. Build sklearn Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**config.TFIDF_PARAMS)),
        ("clf",   MultinomialNB()),
    ])

    # 6. Hyperparameter tuning via GridSearchCV + StratifiedKFold
    logger.info("Running GridSearchCV (%d-fold CV)...", config.CV_FOLDS)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=config.PARAM_GRID,
        cv=cv,
        scoring="f1_macro",        # macro F1 is fairer for imbalanced classes
        n_jobs=-1,
        verbose=1,
        refit=True,                # refit best model on full training set
    )
    grid_search.fit(X_train, y_train)

    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_f1  = grid_search.best_score_

    logger.info("Best params : %s", best_params)
    logger.info("Best CV F1  : %.4f", best_cv_f1)

    # 7. Evaluate on held-out test set
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=class_names)

    logger.info("Test Accuracy : %.4f", acc)
    logger.info("\nClassification Report:\n%s", report_str)

    plot_confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    # 8. Cross-validation summary (all folds)
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_summary_path = os.path.join(config.LOG_DIR, "cv_results.csv")
    cv_results.to_csv(cv_summary_path, index=False)
    logger.info("CV results saved → %s", cv_summary_path)

    # 9. Persist model artifacts
    #    We save the vectorizer SEPARATELY so it can be inspected
    #    and swapped independently in future versions.
    vectorizer = best_model.named_steps["tfidf"]
    classifier = best_model.named_steps["clf"]

    save_artifact(vectorizer, config.VECTORIZER_PATH, "TF-IDF vectorizer")
    save_artifact(classifier, config.MODEL_PATH,      "MultinomialNB model")
    save_artifact(le,         config.LABEL_PATH,      "LabelEncoder")

    # 10. Save metadata (human-readable)
    metadata = {
        "trained_at":      datetime.utcnow().isoformat() + "Z",
        "classes":         class_names,
        "n_train":         len(X_train),
        "n_test":          len(X_test),
        "best_params":     best_params,
        "cv_f1_macro":     round(best_cv_f1, 4),
        "test_accuracy":   round(acc, 4),
        "test_f1_macro":   round(report["macro avg"]["f1-score"], 4),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
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
