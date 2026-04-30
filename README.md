# SMS Classifier — FastText

A production-grade text classification system to label SMS / bank messages
as **CREDIT**, **DEBIT**, **UNKNOWN**, or any custom classes you define.

Powered by **Facebook FastText** — fast training, sub-word embeddings that
handle SMS abbreviations & misspellings, and lightweight on-device deployment
via the quantized `.ftz` format.

---

## Project Structure

```
classification-model/
├── config.py          ← All settings in one place
├── preprocess.py      ← Text cleaning (shared by train & inference)
├── data_loader.py     ← Loads labelled .txt files
├── train.py           ← Full FastText training pipeline (run this first)
├── predict.py         ← Load model & predict (single or batch)
├── api.py             ← FastAPI REST server
├── retrain_from_test_logs.py  ← Auto-retrain from test CSV logs
├── data/
│   ├── credit.txt     ← One message per line for CREDIT class
│   ├── debit.txt      ← One message per line for DEBIT class
│   └── unknown.txt    ← One message per line for UNKNOWN class
├── models/            ← Saved model artifacts (auto-created after training)
│   ├── classifier.bin       ← Full FastText model
│   ├── classifier.ftz       ← Quantized/compressed model (for production)
│   ├── metadata.json        ← Training metrics & config
│   ├── train.ft.txt         ← Generated FastText-format training data
│   └── test.ft.txt          ← Generated FastText-format test data
├── test/              ← Model test suite
├── logs/              ← Training logs + plots (auto-created)
```

---

## Quick Start

### 1. Install dependencies
```bash
uv sync
```

### 2. Add your training data
Put your messages (one per line) in the `data/` folder:
- `data/credit.txt`
- `data/debit.txt`
- `data/unknown.txt`

**Recommended: ≥ 200 messages per class for good accuracy.**

### 3. Train the model
```bash
uv run train.py
```
This will:
- Clean and preprocess your text
- Split data into train/test sets (80/20)
- Train a FastText supervised classifier (with optional auto-tuning)
- Evaluate on a held-out test set
- Save `models/classifier.bin` (full) and `models/classifier.ftz` (quantized)
- Save `logs/confusion_matrix.png` and `logs/class_distribution.png`

#### Training modes

**Auto-tune (default):** FastText automatically searches for the best
hyperparameters for the configured duration:
```python
# config.py
AUTOTUNE_DURATION = 300   # seconds (set to 0 to disable)
```

**Manual hyperparameters:** Set `AUTOTUNE_DURATION = 0` and tune
`FASTTEXT_PARAMS` in `config.py`:
```python
FASTTEXT_PARAMS = {
    "lr": 0.5,
    "epoch": 50,
    "wordNgrams": 2,
    "dim": 50,
    "loss": "softmax",
    "minCount": 2,
    "minn": 2,
    "maxn": 5,
}
```

### 4. Test predictions (CLI)
```bash
uv run predict.py "message text here"
```

### 4.1 Run model test suite (recommended)
Use the labeled files under `test/data/` to run an end-to-end evaluation
against the saved model artifacts.

```bash
uv run test/run_model_test.py
```

This script:
- Loads labeled test messages from `test/data/credit.txt`, `test/data/debit.txt`, and `test/data/unknow.txt` (or `unknown.txt`)
- Runs batch prediction with the current trained model
- Calculates overall accuracy and per-label accuracy
- Writes result files into `test/log/`

Generated test outputs:
- `test/log/model_test_predictions.csv` (row-wise predictions)
- `test/log/model_test_summary.json` (accuracy + metadata summary)
- `test/log/model_test_confusion_matrix.png` (confusion matrix image)

Example output:
```text
Model test complete
Input dir : /home/.../test/data
Output dir: /home/.../test/log
Accuracy  : 100.00%
Uncertain : 0 / 8
CSV log   : /home/.../test/log/model_test_predictions.csv
JSON log  : /home/.../test/log/model_test_summary.json
Matrix    : /home/.../test/log/model_test_confusion_matrix.png
```

### 5. Start the REST API
```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## REST API Endpoints

| Method | Endpoint         | Description                   |
|--------|------------------|-------------------------------|
| GET    | `/health`        | Model info + readiness check  |
| POST   | `/predict`       | Classify a single message     |
| POST   | `/predict/batch` | Classify up to 100 messages   |
| GET    | `/model/info`    | Full training metadata        |
| GET    | `/docs`          | Auto-generated Swagger UI     |

### Example: Single prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Rs.5000 debited from your account XX1234"}'
```
**Response:**
```json
{
  "text": "Rs.5000 debited from your account XX1234",
  "predicted_class": "DEBIT",
  "confidence": 0.9712,
  "is_uncertain": false,
  "all_scores": {"CREDIT": 0.02, "DEBIT": 0.97, "UNKNOWN": 0.01},
  "latency_ms": 0.3
}
```

### Example: Batch prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["credit received", "amount debited", "your OTP is 1234"]}'
```

---

## Adding New Classes

1. Create a new file in `data/`, e.g. `data/fraud.txt`
2. Add one line to `config.py`:
   ```python
   DATA_FILES = {
       "CREDIT":  "credit.txt",
       "DEBIT":   "debit.txt",
       "UNKNOWN": "unknown.txt",
       "FRAUD":   "fraud.txt",   # ← new class
   }
   ```
3. Re-run `python train.py` — the pipeline handles the rest automatically.

---

## Configuration Reference (`config.py`)

| Setting                | Default  | Description                                     |
|------------------------|----------|-------------------------------------------------|
| `TEST_SIZE`            | 0.20     | Fraction of data held out for testing           |
| `CONFIDENCE_THRESHOLD` | 0.60     | Below this → returns `UNCERTAIN`                |
| `AUTOTUNE_DURATION`    | 300      | Seconds for auto-tuning (0 = use manual params) |
| `FASTTEXT_PARAMS`      | see file | Manual hyperparameters (used when autotune=0)   |

---

## Model Artifacts

| File | Description |
|---|---|
| `classifier.bin` | Full FastText model (~2–5 MB). Use for retraining/inspection. |
| `classifier.ftz` | Quantized model (~0.5–1 MB). Use for production & on-device. |
| `metadata.json` | Training metrics, class list, and configuration snapshot. |
| `train.ft.txt` | FastText-format training data (auto-generated). |
| `test.ft.txt` | FastText-format test data (auto-generated). |

---

## Improving Accuracy

| Action                                  | Expected Gain  |
|-----------------------------------------|----------------|
| Add more data (≥ 500/class)             | High           |
| Balance classes (equal samples each)    | Medium–High    |
| Enable auto-tune (AUTOTUNE_DURATION>0)  | Medium–High    |
| Increase epoch count                    | Low–Medium     |
| Use wordNgrams=3 (trigrams)             | Low–Medium     |
| Add domain-specific stopwords           | Medium         |
