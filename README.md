# Credit Card Fraud Detection

Fintech Course Project - Duc Linh Pham
Deadline: 06/05/2026

---

## 📊 Dataset

The dataset used in this project is publicly available on Kaggle:

🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- Contains anonymized transaction data
- Highly imbalanced dataset (fraud cases are rare)
- Features include PCA-transformed variables (V1–V28), Time, and Amount

---

## How To Run Demo In 2 Commands

Assumption: dependencies are installed and model artifact `src/best_rf_model.pkl` exists.

1. Start API server:

```bash
./venv/bin/python deployment/flask_api.py
```

2. Run prediction request (open another terminal):

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Expected response format:

```json
{
  "prediction": 0,
  "fraud_probability": 0.0,
  "risk_level": "LOW"
}
```

Actual demo run (validated):

```json
{
  "fraud_probability": 0.0,
  "prediction": 0,
  "risk_level": "LOW"
}
```

```json
{
  "feature_count": 30,
  "model_path": "src/best_rf_model.pkl",
  "status": "Fraud Detection API v1.0 - Random Forest Champion"
}
```

---

## Final Submission Checklist

- Detailed checklist for Word/PDF submission: `reports/final_submission_checklist.md`
