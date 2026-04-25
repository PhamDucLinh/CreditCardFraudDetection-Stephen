# Final Submission Checklist (Word/PDF)

## 1) Code Artifacts
- [ ] `notebooks/01_full_eda.py` (EDA script)
- [ ] `notebooks/02_models.py` (model training + save RF artifact)
- [ ] `deployment/flask_api.py` (API endpoints `/health`, `/predict`)
- [ ] `src/best_rf_model.pkl` (production model artifact)
- [ ] `sample_request.json` (quick API request payload)

## 2) Report Artifacts
- [ ] `reports/model_comparison.csv` (metrics table for Word)
- [ ] EDA charts (`reports/01_*.png` -> `reports/06_*.png`)
- [ ] Model charts (`reports/07_roc_comparison.png`, `reports/08_xgb_feature_importance.png`)
- [ ] Progress log `reports/project_timeline.md`

## 3) Word Report Content
- [ ] Problem statement + business impact
- [ ] Dataset overview + class imbalance analysis
- [ ] EDA key findings (with figures)
- [ ] Model comparison (Precision/Recall/F1/ROC-AUC table)
- [ ] Final model selection rationale (Random Forest champion)
- [ ] API deployment flow + sample request/response
- [ ] Conclusion + future improvements

## 4) PDF Export
- [ ] Export Word report to PDF
- [ ] Verify figures/tables render correctly
- [ ] Verify section numbering and references

## 5) Demo Readiness
- [ ] Start server: `./venv/bin/python deployment/flask_api.py`
- [ ] Test health: `curl http://localhost:5000/health`
- [ ] Test predict: `curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @sample_request.json`
- [ ] Screenshot terminal outputs for appendix (optional)

## 6) Final Package
- [ ] Include code + report files in final folder/zip
- [ ] Remove unnecessary temp files
- [ ] Double-check Git commit history and final branch status
