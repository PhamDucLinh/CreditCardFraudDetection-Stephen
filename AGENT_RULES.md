# 🚀 CREDIT CARD FRAUD DETECTION AGENT RULES

## 🎯 PROJECT GOAL: F1-score >0.85 trên Kaggle dataset (284K transactions, 0.17% fraud)

## 📋 CORE OUTPUTS BẮT BUỘC:

reports/report_final.docx (15 trang): Intro → EDA → Models → Evaluation → Deployment

notebooks/01_eda.ipynb: Stats + 7 charts (pie fraud, hist Amount/Time, corr heatmap)

notebooks/02_models.ipynb: Logistic + RF + XGBoost + SMOTE + F1/ROC metrics table

src/models.py: Production-ready class với predict() method

deployment/flask_api.py: Real-time fraud API endpoint

requirements.txt: Exact versions đã install

README.md: Run instructions + results summary

text

## 📁 PROJECT STRUCTURE:

fraud_detection/
├── data/ # creditcard.csv (284807 x 31)
├── notebooks/ # 01_eda.py, 02_models.py, 03_deployment.py
├── reports/ # report_final.docx + 10+ charts PNG
├── src/ # models.py, utils.py
├── deployment/ # flask_api.py
├── AGENT_RULES.md # File này
├── requirements.txt
└── README.md

text

## 🛠️ CODING RULES:

✅ ALWAYS use venv: source venv/bin/activate
✅ SMOTE(random_state=42) cho imbalanced data (0.172% fraud)
✅ Metrics ưu tiên: RECALL > PRECISION > F1 > ROC-AUC
✅ Models sequence:

1. LogisticRegression(class_weight='balanced') - baseline
2. RandomForestClassifier(class_weight='balanced', n_estimators=100)
3. XGBClassifier(scale_pos_weight=500, random_state=42) - best
   ✅ Data split: train_test_split(test_size=0.2, stratify=y, random_state=42)
   ✅ Scale CHỈ Amount: StandardScaler().fit_transform(df[['Amount']])
   ✅ Plots: plt.savefig('reports/[name].png', dpi=300, bbox_inches='tight')
   ✅ Comments: Tiếng Việt + English, # ===== SECTION NAME =====
   ✅ PEP8: black . && isort . trước commit
   ✅ Git: git add . && git commit -m "Day X: [summary]" && git push

text

## 📊 EXPECTED METRICS (TARGET):

Logistic: F1 ~0.75 | Recall ~0.85
RandomForest: F1 ~0.82 | Recall ~0.90
XGBoost: F1 >0.85 | Recall >0.95 | ROC-AUC >0.98

text

## 🎛️ 20-DAY TASK BREAKDOWN:

NGÀY 1-2 (16-17/4): ✅ Setup + Intro + EDA pie chart [90% DONE]
NGÀY 3-4 (18-19/4): EDA full (hist, corr, boxplots) + Word phần 2
NGÀY 5-8 (20-23/4): 3 models + SMOTE + baseline metrics table
NGÀY 9-12 (24-27/4): Hyperparam tuning + ROC curves + model comparison
NGÀY 13-15 (28-30/4): SHAP explainability + deployment proposal
NGÀY 16-18 (1-3/5): Word report hoàn chỉnh + Flask API demo
NGÀY 19-20 (4-5/5): Final test + Zalo submit 06/05/2026

text

## 💬 PROMPT PATTERNS (Copy-paste ready):

DATA: "Load data/creditcard.csv, show fraud rate pie chart reports/fraud_pie.png"
EDA: "EDA creditcard.csv: hist Amount/Time by Class, corr heatmap top 10"
MODEL: "Train XGBoost fraud detection SMOTE balanced, F1/ROC table vs Logistic/RF"
TUNE: "GridSearchCV XGBoost max_depth=3-6, learning_rate=0.01-0.1"
DEPLOY: "Flask API /predict endpoint nhận JSON transaction → fraud probability"
BUGFIX: "Debug [error message], theo project rules"
SUMMARY: "Timeline progress? Next 3 tasks? Word report status?"

text

## ❌ ABSOLUTE NOs:

❌ Train full dataset (NO cross_val_score trên raw data)
❌ Quên SMOTE() → F1 <0.5 vì imbalanced
❌ Scale tất cả features (V1-V28 đã PCA ready)
❌ print() thay savefig()
❌ Random_state khác 42
❌ Không git commit daily
❌ Deadline miss 06/05/2026

text

## 🔧 GIT WORKFLOW MỖI NGÀY:

```bash
git add .
git commit -m "Day X: [pie chart/EDA/models/F1 0.87/etc]"
git push origin main
```

## 📈 SUCCESS CRITERIA:

✅ [] Fraud Recall >95% (bắt hết fraud)
✅ [] Test F1-score >0.85 (balance precision/recall)
✅ [] ROC-AUC >0.98
✅ [] Word 15+ pages, 10+ charts
✅ [] Flask API demo: curl -X POST /predict
✅ [] Zalo submit 05/05/2026 trước 23:59

text

## 🤖 AGENT BEHAVIOR:

✅ ALWAYS confirm: "✅ Task X completed. Output saved: [path]. Next: [step]"
✅ BEFORE coding: "Confirm theo rules? SMOTE? Metrics? Save plot?"
✅ ERROR handling: "Bug [error]. Fix: [solution]. Rerun command:"
✅ Progress tracking: "Day X/20 complete. Y% done. Next 3 tasks:"

text

---
