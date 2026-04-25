# ===== 02_MODELS_TRAINING.py - NGÀY 5-8 =====
# Credit Card Fraud | Duc Linh Pham | 21/04/2026
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import os
import warnings
warnings.filterwarnings('ignore')

# Setup
os.makedirs('reports', exist_ok=True)
os.makedirs('src', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

print("🚀 MODEL TRAINING STARTING...")

# ===== LOAD & PREPARE DATA =====
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Scale Amount only (V1-V28 already PCA)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X['Amount_scaled'] = scaler.fit_transform(X[['Amount']])
X = X.drop('Amount', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)
print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean():.4%}")

# ===== SMOTE BALANCE =====
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"✅ SMOTE balanced: {y_train_bal.mean():.4%} fraud")

# ===== 1. LOGISTIC REGRESSION (BASELINE) =====
lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(X_train_bal, y_train_bal)
lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:,1]
lr_f1 = classification_report(y_test, lr_pred, output_dict=True)['1']['f1-score']

print(f"Logistic F1: {lr_f1:.3f}")

# ===== 2. RANDOM FOREST =====
rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, 
                           random_state=42, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]
rf_f1 = classification_report(y_test, rf_pred, output_dict=True)['1']['f1-score']

print(f"Random Forest F1: {rf_f1:.3f}")

# ===== 3. XGBOOST (BEST MODEL) =====
xgb_model = XGBClassifier(scale_pos_weight=500, random_state=42, 
                         n_estimators=200, learning_rate=0.1, n_jobs=-1)
xgb_model.fit(X_train_bal, y_train_bal)
xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:,1]
xgb_f1 = classification_report(y_test, xgb_pred, output_dict=True)['1']['f1-score']

print(f"✅ XGBoost F1: {xgb_f1:.3f}")

# ===== METRICS TABLE =====
results = pd.DataFrame({
    'Model': ['Logistic', 'Random Forest', 'XGBoost'],
    'Precision': [classification_report(y_test, lr_pred, output_dict=True)['1']['precision'],
                  classification_report(y_test, rf_pred, output_dict=True)['1']['precision'],
                  classification_report(y_test, xgb_pred, output_dict=True)['1']['precision']],
    'Recall': [classification_report(y_test, lr_pred, output_dict=True)['1']['recall'],
               classification_report(y_test, rf_pred, output_dict=True)['1']['recall'],
               classification_report(y_test, xgb_pred, output_dict=True)['1']['recall']],
    'F1-Score': [lr_f1, rf_f1, xgb_f1],
    'ROC-AUC': [roc_auc_score(y_test, lr_prob), roc_auc_score(y_test, rf_prob), roc_auc_score(y_test, xgb_prob)]
})

print("\n📊 FINAL METRICS TABLE:")
print(results.round(3).to_markdown())

# Save table for Word
results.to_csv('reports/model_comparison.csv', index=False)
print("✅ Saved: reports/model_comparison.csv")

# ===== ROC CURVES COMPARISON =====
plt.figure(figsize=(10,8))
for model, prob, name in zip([lr, rf, xgb_model], [lr_prob, rf_prob, xgb_prob], 
                           ['Logistic', 'Random Forest', 'XGBoost']):
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, prob):.3f})')

plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('reports/07_roc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Chart 7: roc_comparison.png")

# ===== FEATURE IMPORTANCE (XGBoost) =====
xgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(data=xgb_importance, x='importance', y='feature')
plt.title('Top 10 XGBoost Feature Importance')
plt.savefig('reports/08_xgb_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Chart 8: xgb_feature_importance.png")

# ===== SAVE PRODUCTION RANDOM FOREST =====
rf_artifact = {
    'model': rf,
    'scaler': scaler,
    'feature_columns': X.columns.tolist(),
    'threshold': 0.5,
    'metrics': {
        'f1': float(rf_f1),
        'roc_auc': float(roc_auc_score(y_test, rf_prob))
    }
}
joblib.dump(rf_artifact, 'src/best_rf_model.pkl')
print("✅ Production model saved: src/best_rf_model.pkl")

print("\n🎉 MODEL TRAINING COMPLETE!")
print("📁 Outputs: model_comparison.csv + 2 charts")
print("✅ Next: Ngày 9-12 Hyperparameter tuning + SHAP")
