# =============================================================================
# config.py — Central configuration for the SMS classifier (FastText)
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

MODEL_PATH           = os.path.join(MODEL_DIR, "classifier.bin")
MODEL_QUANTIZED_PATH = os.path.join(MODEL_DIR, "classifier.ftz")
METADATA_PATH        = os.path.join(MODEL_DIR, "metadata.json")

# Intermediate FastText-format files (generated during training)
FASTTEXT_TRAIN_FILE = os.path.join(MODEL_DIR, "train.ft.txt")
FASTTEXT_TEST_FILE  = os.path.join(MODEL_DIR, "test.ft.txt")

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

# ── FastText Hyperparameters ─────────────────────────────────────────────────
FASTTEXT_PARAMS = {
    "lr":         0.5,         # learning rate
    "epoch":      50,          # number of training epochs
    "wordNgrams": 2,           # use bigrams (like old TF-IDF ngram_range=(1,2))
    "dim":        50,          # embedding dimension
    "loss":       "softmax",   # loss function (softmax for multi-class)
    "minCount":   2,           # min word frequency (like old min_df=2)
    "minn":       2,           # min char n-gram length (sub-word features)
    "maxn":       5,           # max char n-gram length
    "bucket":     200000,      # number of hash buckets for char n-grams
}

# ── Auto-Tune ────────────────────────────────────────────────────────────────
# Set to 0 to disable auto-tuning and use manual FASTTEXT_PARAMS instead.
# When enabled, FastText will search for optimal hyperparameters for the given
# duration (in seconds). Recommended: 120–600 for thorough search.
AUTOTUNE_DURATION = 10

# ── Quantization Search (size reduction with accuracy guardrail) ────────────
# The trainer will try these candidates and keep the smallest .ftz model
# whose test accuracy is >= (baseline_accuracy - QUANTIZE_ACCURACY_TOLERANCE).
QUANTIZE_ACCURACY_TOLERANCE = 0.0
QUANTIZE_CANDIDATES = [
    # Baseline quantization behavior (closest to current pipeline)
    {"retrain": True, "qnorm": False, "cutoff": None,  "dsub": None},
    # Often smaller with little/no loss
    {"retrain": True, "qnorm": True,  "cutoff": 80000, "dsub": 2},
    # More aggressive compression
    {"retrain": True, "qnorm": True,  "cutoff": 50000, "dsub": 4},
    # Strong compression candidate
    {"retrain": True, "qnorm": True,  "cutoff": 30000, "dsub": 4},
]

# ── Training ──────────────────────────────────────────────────────────────────
TEST_SIZE      = 0.20    # 20 % held-out test set
RANDOM_STATE   = 42
CV_FOLDS       = 5       # stratified k-fold cross-validation

# ── Prediction ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60   # below this → return "UNCERTAIN"

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
