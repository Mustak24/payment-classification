# =============================================================================
# predict.py — Load saved model and run inference
# =============================================================================
import os
import sys
import json
import pickle
import logging
from dataclasses import dataclass, asdict

import numpy as np

import config
from preprocess import clean

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    text:            str
    predicted_class: str
    confidence:      float
    is_uncertain:    bool
    all_scores:      dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


# ── Classifier ────────────────────────────────────────────────────────────────

class SMSClassifier:
    """
    Thread-safe, singleton-style classifier.
    Load once, call .predict() or .predict_batch() many times.
    """

    _instance = None   # module-level cache

    def __init__(self) -> None:
        self._vectorizer  = None
        self._model       = None
        self._label_enc   = None
        self._metadata    = {}
        self._loaded      = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> "SMSClassifier":
        """Load all saved artifacts from disk. Call once at startup."""
        for path, name in [
            (config.MODEL_PATH,      "model"),
            (config.VECTORIZER_PATH, "vectorizer"),
            (config.LABEL_PATH,      "label encoder"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {name}: {path}\n"
                    "Run  python train.py  first to train and save the model."
                )

        with open(config.VECTORIZER_PATH, "rb") as f:
            self._vectorizer = pickle.load(f)
        with open(config.MODEL_PATH, "rb") as f:
            self._model = pickle.load(f)
        with open(config.LABEL_PATH, "rb") as f:
            self._label_enc = pickle.load(f)

        if os.path.exists(config.METADATA_PATH):
            with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

        self._loaded = True
        logger.info(
            "Model loaded. Classes: %s | Vocab: %s | Trained: %s",
            self._metadata.get("classes", "?"),
            self._metadata.get("vocabulary_size", "?"),
            self._metadata.get("trained_at", "?"),
        )
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call SMSClassifier().load() before predicting.")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, text: str, threshold: float | None = None) -> PredictionResult:
        """
        Predict the class of a single message.

        Args:
            text:      Raw SMS / message string.
            threshold: Confidence threshold. Overrides config if provided.
                       Predictions below this return 'UNCERTAIN'.

        Returns:
            PredictionResult dataclass.
        """
        self._ensure_loaded()
        threshold = threshold if threshold is not None else config.CONFIDENCE_THRESHOLD

        cleaned  = clean(text)
        vec      = self._vectorizer.transform([cleaned])
        proba    = self._model.predict_proba(vec)[0]           # shape (n_classes,)
        class_names = self._label_enc.classes_

        all_scores = {cls: round(float(p), 4) for cls, p in zip(class_names, proba)}

        best_idx    = int(np.argmax(proba))
        confidence  = float(proba[best_idx])
        predicted   = class_names[best_idx]
        uncertain   = confidence < threshold

        return PredictionResult(
            text=text,
            predicted_class="UNCERTAIN" if uncertain else predicted,
            confidence=round(confidence, 4),
            is_uncertain=uncertain,
            all_scores=all_scores,
        )

    def predict_batch(
        self,
        texts: list[str],
        threshold: float | None = None,
    ) -> list[PredictionResult]:
        """
        Predict for a list of messages efficiently (single vectorize call).
        """
        self._ensure_loaded()
        threshold = threshold if threshold is not None else config.CONFIDENCE_THRESHOLD

        cleaned     = [clean(t) for t in texts]
        vecs        = self._vectorizer.transform(cleaned)
        probas      = self._model.predict_proba(vecs)          # shape (n, n_classes)
        class_names = self._label_enc.classes_

        results = []
        for text, proba in zip(texts, probas):
            all_scores  = {cls: round(float(p), 4) for cls, p in zip(class_names, proba)}
            best_idx    = int(np.argmax(proba))
            confidence  = float(proba[best_idx])
            predicted   = class_names[best_idx]
            uncertain   = confidence < threshold

            results.append(PredictionResult(
                text=text,
                predicted_class="UNCERTAIN" if uncertain else predicted,
                confidence=round(confidence, 4),
                is_uncertain=uncertain,
                all_scores=all_scores,
            ))

        return results

    @property
    def metadata(self) -> dict:
        return self._metadata


# ── Module-level singleton ────────────────────────────────────────────────────

def get_classifier() -> SMSClassifier:
    """Return a loaded SMSClassifier singleton (load once per process)."""
    if SMSClassifier._instance is None:
        SMSClassifier._instance = SMSClassifier().load()
    return SMSClassifier._instance


# ── CLI quick-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    clf = get_classifier()

    test_messages = sys.argv[1:] if len(sys.argv) > 1 else [
        """"X89 your account has been credited Rs5000. of Proceed to withdraw during available times. Lots redeeming! bit.ly/3Qfe4sl JwelryGarmentsAcesrynMore"""
    ]

    print("\n" + "=" * 65)
    print("  SMS Classifier — Prediction Demo")
    print("=" * 65)

    for msg in test_messages:
        result = clf.predict(msg)
        flag = "⚠ UNCERTAIN" if result.is_uncertain else ""
        print(f"\nMessage   : {result.text[:80]}")
        print(f"Predicted : {result.predicted_class}  (conf={result.confidence:.2%})  {flag}")
        print(f"All scores: {result.all_scores}")
