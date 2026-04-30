# =============================================================================
# predict.py — Load saved FastText model and run inference
# =============================================================================
import os
import sys
import json
import logging
from dataclasses import dataclass, asdict

import numpy as np
import fasttext

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


# ── FastText numpy 2.x compatibility ─────────────────────────────────────────

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


# ── Classifier ────────────────────────────────────────────────────────────────

class SMSClassifier:
    """
    Thread-safe, singleton-style classifier backed by FastText.
    Load once, call .predict() or .predict_batch() many times.
    """

    _instance = None   # module-level cache

    def __init__(self) -> None:
        self._model      = None
        self._class_names: list[str] = []
        self._metadata:  dict = {}
        self._loaded     = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> "SMSClassifier":
        """Load a saved FastText model from disk. Call once at startup."""
        # Prefer the quantized model (.ftz) if available; fall back to full (.bin)
        model_path = config.MODEL_QUANTIZED_PATH
        if not os.path.exists(model_path):
            model_path = config.MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing FastText model: tried {config.MODEL_QUANTIZED_PATH} "
                f"and {config.MODEL_PATH}\n"
                "Run  python train.py  first to train and save the model."
            )

        self._model = fasttext.load_model(model_path)

        # Derive class names from the model's labels (__label__CREDIT → CREDIT)
        self._class_names = sorted(
            label.replace("__label__", "") for label in self._model.labels
        )

        if os.path.exists(config.METADATA_PATH):
            with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

        self._loaded = True
        logger.info(
            "FastText model loaded from %s. Classes: %s | Trained: %s",
            model_path,
            self._class_names,
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

        cleaned = clean(text)

        # Predict all classes so we can return full score distribution
        k = len(self._class_names)
        labels, probs = self._model.predict(cleaned, k=k)

        all_scores = {
            label.replace("__label__", ""): round(float(prob), 4)
            for label, prob in zip(labels, probs)
        }

        best_label  = labels[0].replace("__label__", "")
        confidence  = float(probs[0])
        uncertain   = confidence < threshold

        return PredictionResult(
            text=text,
            predicted_class="UNCERTAIN" if uncertain else best_label,
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
        Predict for a list of messages.

        FastText prediction is extremely fast (microseconds per call),
        so a simple loop is more than adequate.
        """
        self._ensure_loaded()
        threshold = threshold if threshold is not None else config.CONFIDENCE_THRESHOLD

        results: list[PredictionResult] = []
        k = len(self._class_names)

        for text in texts:
            cleaned = clean(text)
            labels, probs = self._model.predict(cleaned, k=k)

            all_scores = {
                label.replace("__label__", ""): round(float(prob), 4)
                for label, prob in zip(labels, probs)
            }

            best_label  = labels[0].replace("__label__", "")
            confidence  = float(probs[0])
            uncertain   = confidence < threshold

            results.append(PredictionResult(
                text=text,
                predicted_class="UNCERTAIN" if uncertain else best_label,
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
        """"₹50 cashback till 11 AM! 1:28:55 Scan & Pay for anything above ₹50 with BHIM and get upto ₹50 cashback. Thursday Cashback hour ends at 11 AM  Pay Now"""
    ]

    print("\n" + "=" * 65)
    print("  SMS Classifier (FastText) — Prediction Demo")
    print("=" * 65)

    for msg in test_messages:
        result = clf.predict(msg)
        flag = "⚠ UNCERTAIN" if result.is_uncertain else ""
        print(f"\nMessage   : {result.text[:80]}")
        print(f"Predicted : {result.predicted_class}  (conf={result.confidence:.2%})  {flag}")
        print(f"All scores: {result.all_scores}")
