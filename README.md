# SMS Classifier — Production Ready

A production-grade text classification system to label SMS / bank messages
as **CREDIT**, **DEBIT**, **UNKNOWN**, or any custom classes you define.

---

## Project Structure

```
sms_classifier/
├── config.py          ← All settings in one place
├── preprocess.py      ← Text cleaning (shared by train & inference)
├── data_loader.py     ← Loads labelled .txt files
├── train.py           ← Full training pipeline (run this first)
├── predict.py         ← Load model & predict (single or batch)
├── api.py             ← FastAPI REST server
├── requirements.txt
├── data/
│   ├── credit.txt     ← One message per line for CREDIT class
│   ├── debit.txt      ← One message per line for DEBIT class
│   └── unknown.txt    ← One message per line for UNKNOWN class
├── models/            ← Saved model artifacts (auto-created after training)
├── logs/              ← Training logs + plots (auto-created)
└── tests/
    └── test_classifier.py
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your training data
Put your messages (one per line) in the `data/` folder:
- `data/credit.txt`
- `data/debit.txt`
- `data/unknown.txt`

**Recommended: ≥ 200 messages per class for good accuracy.**

### 3. Train the model
```bash
python train.py
```
This will:
- Clean and vectorize your text
- Run 5-fold cross-validation with hyperparameter tuning
- Evaluate on a held-out test set
- Save `models/classifier.pkl`, `models/vectorizer.pkl`, `models/label_encoder.pkl`
- Save `logs/confusion_matrix.png` and `logs/class_distribution.png`

### 4. Test predictions (CLI)
```bash
python predict.py "message text here"
```

### 5. Start the REST API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 6. Run tests
```bash
pytest tests/ -v
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
  "latency_ms": 1.8
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

| Setting                | Default  | Description                                   |
|------------------------|----------|-----------------------------------------------|
| `TEST_SIZE`            | 0.20     | Fraction of data held out for testing         |
| `CV_FOLDS`             | 5        | Number of cross-validation folds              |
| `CONFIDENCE_THRESHOLD` | 0.60     | Below this → returns `UNCERTAIN`              |
| `TFIDF_PARAMS`         | see file | Vectorizer hyperparameters                    |
| `PARAM_GRID`           | see file | Hyperparameter search space for GridSearchCV  |

---

## Improving Accuracy

| Action                                  | Expected Gain  |
|-----------------------------------------|----------------|
| Add more data (≥ 500/class)             | High           |
| Balance classes (equal samples each)    | Medium–High    |
| Add domain-specific stopwords           | Medium         |
| Switch to DistilBERT (transformer)      | High (GPU rec.)|
| Add n-gram range (1,3) in TFIDF_PARAMS  | Low–Medium     |
