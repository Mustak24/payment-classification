# =============================================================================
# config.py — Central configuration for the SMS classifier
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

MODEL_PATH      = os.path.join(MODEL_DIR, "classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LABEL_PATH      = os.path.join(MODEL_DIR, "label_encoder.pkl")
METADATA_PATH   = os.path.join(MODEL_DIR, "metadata.json")

# ── Data files  ───────────────────────────────────────────────────────────────
# Map each CLASS LABEL → text file inside DATA_DIR
# Add or remove classes here — the rest of the pipeline picks it up automatically.
DATA_FILES = {
    "CREDIT": "credit.txt",
    "DEBIT":  "debit.txt",
    "UNKNOWN": "unknown.txt",
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
MAX_TEXT_LENGTH = 500          # characters; longer texts are truncated

# ── TF-IDF Vectorizer ────────────────────────────────────────────────────────
TFIDF_PARAMS = {
    "ngram_range": (1, 2),     # unigrams + bigrams
    "max_features": 10_000,
    "sublinear_tf": True,      # apply log normalization
    "min_df": 2,               # ignore terms appearing in fewer than 2 docs
    "max_df": 0.95,            # ignore terms appearing in >95 % of docs
    "strip_accents": "unicode",
}

# ── Model Hyperparameter Search Space ────────────────────────────────────────
# GridSearchCV will try every combination below.
PARAM_GRID = {
    "clf__alpha": [0.01, 0.1, 0.5, 1.0],   # Naive Bayes smoothing
}

# ── Training ──────────────────────────────────────────────────────────────────
TEST_SIZE      = 0.20    # 20 % held-out test set
RANDOM_STATE   = 42
CV_FOLDS       = 5       # stratified k-fold cross-validation

# ── Prediction ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60   # below this → return "UNCERTAIN"

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
