# Day 09 - Upgrade XGBoost (F1 > 0.85 Target)

## 1) Mục tiêu ngày
- Nâng chất lượng mô hình XGBoost cho bài toán phát hiện gian lận thẻ tín dụng.
- Mục tiêu kỳ vọng: F1-score lớp gian lận (`Class=1`) > 0.85.
- Thực hiện tuning bằng `GridSearchCV` trên dữ liệu đã balance bằng SMOTE.

## 2) Đầu vào
- Dataset: `data/creditcard.csv`
- Baseline script: `notebooks/02_models.py`
- Baseline metrics (sau khi rerun `02_models.py`):

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic | 0.118 | 0.908 | 0.209 | 0.976 |
| Random Forest | 0.827 | 0.827 | 0.827 | 0.969 |
| XGBoost | 0.461 | 0.898 | 0.609 | 0.971 |

## 3) Công việc thực hiện
### Bước 1: Kiểm tra môi trường và chạy lại baseline
- Kiểm tra Python/pip trong `venv`.
- Xác nhận `tabulate` đã có trong `venv`.
- Chạy lại `notebooks/02_models.py` để lấy bảng F1 đầy đủ.

**Đầu vào:** `notebooks/02_models.py`, `data/creditcard.csv`  
**Đầu ra:** `reports/model_comparison.csv`, `reports/07_roc_comparison.png`, `reports/08_xgb_feature_importance.png`

### Bước 2: Tạo script tuning mới
- Tạo file `notebooks/03_xgb_tuning.py`.
- Pipeline trong script:
  - Load dữ liệu và tách `X/y`.
  - Scale cột `Amount` thành `Amount_scaled`.
  - Chia train/test có `stratify`.
  - Dùng SMOTE cho tập train.
  - GridSearchCV cho XGBoost với `scoring='f1'`, `cv=3`, `n_jobs=-1`.

**Param grid sử dụng:**
```python
{
    'n_estimators': [100, 200],
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'scale_pos_weight': [300, 500],
}
```

**Đầu vào:** `data/creditcard.csv`  
**Đầu ra:** `notebooks/03_xgb_tuning.py`

### Bước 3: Chạy tuning và lưu artifacts
- Chạy `./venv/bin/python notebooks/03_xgb_tuning.py`.
- Script sẽ lưu:
  - `reports/day09_xgb_grid_results.csv`
  - `reports/day09_xgb_classification_report.csv`
  - `reports/day09_xgb_best_params.txt`

**Đầu vào:** `notebooks/03_xgb_tuning.py`  
**Đầu ra:** các file kết quả trong `reports/`

## 4) Kết quả tuning
- Đã chạy GridSearch đầy đủ: `48 fits`.
- Kết quả GridSearch (SMOTE train):
  - `Best params`: `{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200, 'scale_pos_weight': 300}`
  - `Best F1 CV`: `0.9943`
  - `Test F1 (Class=1)`: `0.1901` (không đạt mục tiêu do overfit trên dữ liệu đã cân bằng)
- Upgrade fix (không dùng SMOTE, giữ train gốc):
  - Params: `{'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'scale_pos_weight': 1}`
  - `Final Test F1 (Class=1)`: `0.8541` ✅ đạt mục tiêu `> 0.85`
- Classification report (final model):
  - Precision fraud: `0.9080`
  - Recall fraud: `0.8061`
  - F1 fraud: `0.8541`

## 5) File tạo/chỉnh sửa trong Day 9
- `notebooks/03_xgb_tuning.py` (mới)
- `reports/day09_xgb_tuning.md` (mới)
- `reports/day09_xgb_grid_results.csv` (mới)
- `reports/day09_xgb_classification_report.csv` (mới)
- `reports/day09_xgb_best_params.txt` (mới)

## 6) Ghi chú
- Baseline hiện tại: Random Forest đang tốt nhất với F1 = 0.827.
- Sau Day 9: XGBoost đã vượt baseline và đạt mục tiêu F1 > 0.85 (F1 = 0.8541).
- Insight chính: kết hợp `SMOTE + scale_pos_weight` lớn làm kết quả CV rất cao nhưng tổng quát hóa kém trên test thực tế.
