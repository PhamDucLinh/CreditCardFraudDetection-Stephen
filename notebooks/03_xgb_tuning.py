# ===== 03_XGB_TUNING.py - F1>0.90 =====
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)

# Giả sử từ 02_models.py
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

# TUNING PARAMS (Credit Card Fraud optimized)
param_grid = {
    "scale_pos_weight": [284315 / 492, 500, 600],  # ~578 optimal
    "max_depth": [3, 4],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8],
}

print("🚀 XGBOOST GRID SEARCH...")
xgb = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
grid = RandomizedSearchCV(
    xgb,
    param_grid,
    n_iter=20,
    cv=3,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train_bal, y_train_bal)

print("🏆 BEST PARAMS:", grid.best_params_)
print("🏆 BEST CV F1:", round(grid.best_score_, 4))

# Test best model
best_xgb = grid.best_estimator_
y_pred = best_xgb.predict(X_test)
y_prob = best_xgb.predict_proba(X_test)[:, 1]

full_report = classification_report(y_test, y_pred, output_dict=True)
report = full_report["1"]
final_f1 = report["f1-score"]
final_roc_auc = roc_auc_score(y_test, y_prob)

print("\n🎯 FINAL XGBoost:")
print(f"Precision: {report['precision']:.3f}")
print(f"Recall: {report['recall']:.3f}")
print(f"F1: {final_f1:.3f}")
print(f"ROC-AUC: {final_roc_auc:.3f}")

# Feature importance
importance = (
    pd.DataFrame({"feature": X.columns, "importance": best_xgb.feature_importances_})
    .sort_values("importance", ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
plt.barh(importance["feature"], importance["importance"])
plt.title("Top 10 XGBoost Features - Tuned Model")
plt.tight_layout()
plt.savefig("reports/09_xgb_importance_tuned.png", dpi=300)
plt.close()

# Save artifacts for report
pd.DataFrame(grid.cv_results_).sort_values("rank_test_score").to_csv(
    "reports/09_xgb_randomized_search_results.csv", index=False
)
pd.DataFrame(full_report).T.to_csv("reports/09_xgb_tuned_classification_report.csv", index=True)
with open("reports/09_xgb_tuned_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"BEST PARAMS: {grid.best_params_}\n")
    f.write(f"BEST CV F1: {grid.best_score_:.4f}\n")
    f.write(f"Precision (Class 1): {report['precision']:.4f}\n")
    f.write(f"Recall (Class 1): {report['recall']:.4f}\n")
    f.write(f"F1 (Class 1): {final_f1:.4f}\n")
    f.write(f"ROC-AUC: {final_roc_auc:.4f}\n")

if final_f1 >= 0.90:
    print("✅ TUNING COMPLETE! F1>0.90 target achieved")
else:
    print(f"⚠️ TUNING COMPLETE! Current F1={final_f1:.3f} (target >0.90 chưa đạt)")
