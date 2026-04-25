# ===== 04_XGB_FIXED.py - Anti-overfit F1>0.90 =====
import os
import warnings
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
os.makedirs("src", exist_ok=True)

# Data từ trước (copy từ 02_models.py)
df = pd.read_csv("data/creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ✅ FIXED XGBOOST (SHALLOW + REGULARIZED)
xgb_fixed = XGBClassifier(
    n_estimators=100,           # Giảm từ 200
    max_depth=3,                # Giảm từ 4 (CRITICAL)
    learning_rate=0.05,         # Giảm từ 0.1
    subsample=0.8,              # 80% data
    colsample_bytree=0.8,       # 80% features
    reg_alpha=0.1,              # L1 regularize
    reg_lambda=1.0,             # L2 regularize
    scale_pos_weight=100,       # Conservative vs 577
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)

xgb_fixed.fit(X_train_bal, y_train_bal)
y_pred = xgb_fixed.predict(X_test)
y_prob = xgb_fixed.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, output_dict=True)["1"]
f1 = report["f1-score"]
roc_auc = roc_auc_score(y_test, y_prob)

print("🎯 FIXED XGBoost:")
print(f"Precision: {report['precision']:.3f}")
print(f"Recall: {report['recall']:.3f}")
print(f"F1: {f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")

# So sánh với Random Forest
print("\n📊 VS Random Forest (0.827):", "XGBoost WIN!" if f1 > 0.827 else "RF still better")

# Save best model
joblib.dump(xgb_fixed, "src/best_xgb_model.pkl")
print("✅ Saved production model: src/best_xgb_model.pkl")

# Save summary artifact for report
with open("reports/10_xgb_fixed_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Precision: {report['precision']:.4f}\n")
    f.write(f"Recall: {report['recall']:.4f}\n")
    f.write(f"F1: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"Vs RF 0.827: {'XGBoost WIN!' if f1 > 0.827 else 'RF still better'}\n")

print("✅ Saved report summary: reports/10_xgb_fixed_summary.txt")
