# ===== FLASK_API.py - PRODUCTION FRAUD DETECTION =====
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

MODEL_PATH = 'src/best_rf_model.pkl'
DATA_PATH = 'data/creditcard.csv'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing model artifact at {MODEL_PATH}. Run notebooks/02_models.py first."
    )

artifact = joblib.load(MODEL_PATH)

if isinstance(artifact, dict) and 'model' in artifact:
    rf_model = artifact['model']
    scaler = artifact.get('scaler')
    feature_columns = artifact.get('feature_columns', [])
else:
    # Backward-compatible fallback if only model object was saved.
    rf_model = artifact
    scaler = StandardScaler()
    raw_df = pd.read_csv(DATA_PATH)
    scaler.fit(raw_df[['Amount']])
    feature_columns = list(getattr(rf_model, 'feature_names_in_', []))


def _prepare_features(payload_features: dict) -> pd.DataFrame:
    """Build one-row model input with expected columns and safe defaults."""
    df = pd.DataFrame([payload_features])

    # Compute Amount_scaled if model expects it.
    if 'Amount_scaled' in feature_columns:
        amount_series = pd.to_numeric(df.get('Amount', pd.Series([0.0])), errors='coerce').fillna(0.0)
        df['Amount_scaled'] = scaler.transform(pd.DataFrame({'Amount': amount_series})).ravel()

    # Fill missing expected columns with 0.
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    # Keep only training columns in exact order.
    model_df = df[feature_columns].copy()
    model_df = model_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return model_df


@app.route('/predict', methods=['POST'])
def predict_fraud():
    data = request.get_json(silent=True) or {}
    features = data.get('features')

    if not isinstance(features, dict):
        return jsonify({'error': "Payload must include object field 'features'."}), 400

    model_input = _prepare_features(features)
    prob = float(rf_model.predict_proba(model_input)[0, 1])
    prediction = 1 if prob > 0.5 else 0

    return jsonify({
        'prediction': int(prediction),
        'fraud_probability': prob,
        'risk_level': 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.3 else 'LOW'
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'Fraud Detection API v1.0 - Random Forest Champion',
        'model_path': MODEL_PATH,
        'feature_count': len(feature_columns)
    })


if __name__ == '__main__':
    print("🚀 Fraud API starting... Random Forest model online")
    app.run(debug=True, host='0.0.0.0', port=5000)
