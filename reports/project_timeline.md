# Credit Card Fraud Detection - Project Timeline

File này dùng để ghi log toàn bộ tiến độ theo ngày trong 1 nơi duy nhất.

## Day 09 - Upgrade XGBoost (F1 > 0.85 Target)

### 1) Mục tiêu ngày
- Nâng chất lượng mô hình XGBoost cho bài toán phát hiện gian lận thẻ tín dụng.
- Mục tiêu kỳ vọng: F1-score lớp gian lận (`Class=1`) > 0.85.
- Thực hiện tuning bằng `GridSearchCV` trên dữ liệu đã balance bằng SMOTE.

### 2) Đầu vào
- Dataset: `data/creditcard.csv`
- Baseline script: `notebooks/02_models.py`
- Baseline metrics (sau khi rerun `02_models.py`):

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic | 0.118 | 0.908 | 0.209 | 0.976 |
| Random Forest | 0.827 | 0.827 | 0.827 | 0.969 |
| XGBoost | 0.461 | 0.898 | 0.609 | 0.971 |

### 3) Công việc thực hiện
#### Bước 1: Kiểm tra môi trường và chạy lại baseline
- Kiểm tra Python/pip trong `venv`.
- Xác nhận `tabulate` đã có trong `venv`.
- Chạy lại `notebooks/02_models.py` để lấy bảng F1 đầy đủ.

Đầu vào: `notebooks/02_models.py`, `data/creditcard.csv`  
Đầu ra: `reports/model_comparison.csv`, `reports/07_roc_comparison.png`, `reports/08_xgb_feature_importance.png`

#### Bước 2: Tạo script tuning mới
- Tạo file `notebooks/03_xgb_tuning.py`.
- Pipeline trong script:
  - Load dữ liệu và tách `X/y`.
  - Scale cột `Amount` thành `Amount_scaled`.
  - Chia train/test có `stratify`.
  - Dùng SMOTE cho tập train.
  - GridSearchCV cho XGBoost với `scoring='f1'`, `cv=3`, `n_jobs=-1`.

Param grid sử dụng:

```python
{
    'n_estimators': [100, 200],
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'scale_pos_weight': [300, 500],
}
```

Đầu vào: `data/creditcard.csv`  
Đầu ra: `notebooks/03_xgb_tuning.py`

#### Bước 3: Chạy tuning và lưu artifacts
- Chạy `./venv/bin/python notebooks/03_xgb_tuning.py`.
- Script lưu:
  - `reports/day09_xgb_grid_results.csv`
  - `reports/day09_xgb_classification_report.csv`
  - `reports/day09_xgb_best_params.txt`

Đầu vào: `notebooks/03_xgb_tuning.py`  
Đầu ra: các file kết quả trong `reports/`

### 4) Kết quả
- GridSearch đầy đủ: `48 fits`.
- Kết quả GridSearch (SMOTE train):
  - Best params: `{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200, 'scale_pos_weight': 300}`
  - Best F1 CV: `0.9943`
  - Test F1 (Class=1): `0.1901`
- Upgrade fix (không dùng SMOTE, giữ train gốc):
  - Params: `{'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'scale_pos_weight': 1}`
  - Final Test F1 (Class=1): `0.8541` (đạt mục tiêu `>0.85`)

Classification report (final model):
- Precision fraud: `0.9080`
- Recall fraud: `0.8061`
- F1 fraud: `0.8541`

### 5) File tạo/chỉnh sửa
- `notebooks/03_xgb_tuning.py`
- `reports/day09_xgb_grid_results.csv`
- `reports/day09_xgb_classification_report.csv`
- `reports/day09_xgb_best_params.txt`

### 6) Ghi chú
- Baseline tốt nhất trước Day 9: Random Forest F1 = `0.827`.
- Sau Day 9: XGBoost đạt F1 = `0.8541`.
- Insight chính: kết hợp `SMOTE + scale_pos_weight` lớn cho CV rất cao nhưng tổng quát hóa kém trên test thực tế.

## Day 09-12 - XGBoost Tuning F1>0.90 (RandomizedSearchCV)

### 1) Mục tiêu ngày
- Thực hiện tuning XGBoost theo cấu hình mở rộng để hướng tới mục tiêu F1 `0.92-0.95`.
- Chạy script `notebooks/03_xgb_tuning.py` và ghi nhận best F1 thực tế.

### 2) Đầu vào
- Dataset: `data/creditcard.csv`
- Script tuning: `notebooks/03_xgb_tuning.py`
- Pipeline: chuẩn hóa `Amount`, `train_test_split`, SMOTE trên tập train.

### 3) Công việc thực hiện
#### Bước 1: Cập nhật script theo cấu hình mới
- Chuyển sang `RandomizedSearchCV` với `n_iter=20`, `cv=3`, `scoring='f1'`.
- Param search:
  - `scale_pos_weight`: `[284315/492, 500, 600]`
  - `max_depth`: `[3, 4]`
  - `learning_rate`: `[0.05, 0.1]`
  - `n_estimators`: `[100, 200]`
  - `subsample`: `[0.8, 1.0]`
  - `colsample_bytree`: `[0.8]`

Đầu vào: yêu cầu Day 9-12 từ user  
Đầu ra: script `notebooks/03_xgb_tuning.py` đã cập nhật

#### Bước 2: Chạy tuning
- Lệnh chạy: `./venv/bin/python notebooks/03_xgb_tuning.py`
- Khối lượng chạy: `60 fits` (3 folds x 20 candidates)

Đầu vào: dữ liệu train đã SMOTE  
Đầu ra: best params, best CV F1, test metrics

#### Bước 3: Lưu artifacts
- `reports/09_xgb_importance_tuned.png`
- `reports/09_xgb_randomized_search_results.csv`
- `reports/09_xgb_tuned_classification_report.csv`
- `reports/09_xgb_tuned_summary.txt`

### 4) Kết quả
- Best params:
  - `{'subsample': 1.0, 'scale_pos_weight': 577.8760162601626, 'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1, 'colsample_bytree': 0.8}`
- Best CV F1: `0.9929`
- Test set:
  - Precision (Class 1): `0.0890`
  - Recall (Class 1): `0.8980`
  - F1 (Class 1): `0.1619`
  - ROC-AUC: `0.9832`

### 5) Đánh giá mục tiêu
- Mục tiêu F1 `>0.90`: **chưa đạt** trong cấu hình SMOTE + scale_pos_weight cao.
- Quan sát: mô hình nghiêng mạnh về recall, precision rất thấp nên F1 giảm mạnh trên test set thực tế.

## Day 10 - XGBoost Anti-overfit Attempt (04_xgb_fixed.py)

### 1) Mục tiêu ngày
- Thử cấu hình XGBoost shallow + regularization để giảm overfit từ bản tuning trước.
- Kỳ vọng cải thiện F1 lớp gian lận và vượt baseline Random Forest (0.827).

### 2) Đầu vào
- Dataset: `data/creditcard.csv`
- Baseline tham chiếu: Random Forest F1 = `0.827`
- Cấu hình chạy: `notebooks/04_xgb_fixed.py`

### 3) Công việc thực hiện
#### Bước 1: Tạo script anti-overfit
- Tạo file `notebooks/04_xgb_fixed.py` theo đúng cấu hình đề xuất:
  - `n_estimators=100`
  - `max_depth=3`
  - `learning_rate=0.05`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `reg_alpha=0.1`
  - `reg_lambda=1.0`
  - `scale_pos_weight=100`
- Pipeline dữ liệu: chuẩn hóa `Amount`, train/test split, SMOTE trên tập train.

Đầu vào: yêu cầu cấu hình fix từ user  
Đầu ra: `notebooks/04_xgb_fixed.py`

#### Bước 2: Chạy huấn luyện và đánh giá
- Lệnh chạy: `./venv/bin/python notebooks/04_xgb_fixed.py`
- Script in đầy đủ Precision/Recall/F1/ROC-AUC cho `Class=1`.

Đầu vào: train data sau SMOTE  
Đầu ra: metrics test set

#### Bước 3: Lưu model và artifact
- Save model: `src/best_xgb_model.pkl`
- Save summary: `reports/10_xgb_fixed_summary.txt`

### 4) Kết quả
- Precision: `0.009`
- Recall: `0.959`
- F1: `0.017`
- ROC-AUC: `0.977`
- So với RF (0.827): `RF still better`

### 5) Đánh giá mục tiêu
- Mục tiêu F1 `>0.90`: **không đạt**.
- Quan sát: mô hình bị bias mạnh về dự đoán fraud (recall rất cao nhưng precision gần 0), nên F1 sụt mạnh.

## Day 10-12 - Flask API Production (Random Forest Champion)

### 1) Mục tiêu ngày
- Chốt Random Forest (`F1=0.827`) làm production model.
- Triển khai API dự đoán gian lận realtime với Flask.

### 2) Đầu vào
- Model training script: `notebooks/02_models.py`
- API target file: `deployment/flask_api.py`
- Model artifact cần có: `src/best_rf_model.pkl`

### 3) Công việc thực hiện
#### Bước 1: Lưu Random Forest artifact từ pipeline training
- Cập nhật `notebooks/02_models.py` để save production artifact bằng `joblib`.
- Artifact lưu gồm:
  - `model` (RandomForestClassifier)
  - `scaler` (StandardScaler cho `Amount`)
  - `feature_columns` (thứ tự cột train)
  - `threshold`, `metrics`
- Chạy lại `./venv/bin/python notebooks/02_models.py` để tạo file model.

Đầu vào: `data/creditcard.csv`  
Đầu ra: `src/best_rf_model.pkl`

#### Bước 2: Build Flask API production endpoint
- Cập nhật `deployment/flask_api.py`:
  - `POST /predict`: nhận JSON `features`, chuẩn hóa `Amount`, map đúng schema train, dự đoán `prediction`, `fraud_probability`, `risk_level`.
  - `GET /health`: trả trạng thái service + metadata model.
- API có fallback tương thích nếu artifact chỉ là model object.

Đầu vào: `src/best_rf_model.pkl`  
Đầu ra: API server chạy cổng `5000`

#### Bước 3: Cài dependency và test endpoint
- Cài Flask trong venv: `./venv/bin/pip install flask`
- Cập nhật `requirements.txt` thêm `flask>=3.1.0`
- Test thực tế:
  - `GET /health`
  - `POST /predict` với payload mẫu tối giản (`Time`, `V1`, `V2`, `Amount`)

Đầu vào: local API server  
Đầu ra: JSON response hợp lệ cho cả 2 endpoint

### 4) Kết quả
- Training run xác nhận lại baseline:
  - Random Forest F1: `0.827`
  - Random Forest ROC-AUC: `0.969`
- Artifact đã lưu: `src/best_rf_model.pkl`
- API test kết quả:
  - `GET /health`:
    - `status`: `Fraud Detection API v1.0 - Random Forest Champion`
    - `feature_count`: `30`
  - `POST /predict` (payload mẫu):
    - `prediction`: `0`
    - `fraud_probability`: `0.0`
    - `risk_level`: `LOW`

### 5) File tạo/chỉnh sửa
- `notebooks/02_models.py`
- `deployment/flask_api.py`
- `requirements.txt`
- `src/best_rf_model.pkl`

### 6) Đánh giá
- Random Forest đã sẵn sàng làm model production cho giai đoạn deploy API.
- API hoạt động end-to-end với dữ liệu đầu vào JSON và trả về xác suất gian lận đúng format.

## Day 12 - Final Packaging Round

### 1) Mục tiêu ngày
- Hoàn thiện tài liệu nộp cuối: hướng dẫn demo nhanh + checklist nộp Word/PDF.

### 2) Đầu vào
- Yêu cầu final round từ user:
  - Thêm phần README: chạy demo bằng 2 lệnh
  - Tạo `sample_request.json`
  - Tạo checklist final submission

### 3) Công việc thực hiện
#### Bước 1: Cập nhật README
- Thêm section `How To Run Demo In 2 Commands`.
- Thêm ví dụ response và liên kết checklist nộp cuối.

Đầu vào: `README.md` hiện tại  
Đầu ra: `README.md` có hướng dẫn demo nhanh

#### Bước 2: Tạo payload test mẫu
- Tạo file `sample_request.json` chứa `features` mẫu để gửi trực tiếp vào API bằng `curl -d @sample_request.json`.

Đầu vào: format payload API `/predict`  
Đầu ra: `sample_request.json`

#### Bước 3: Tạo checklist nộp cuối
- Tạo file `reports/final_submission_checklist.md` với checklist:
  - Code artifacts
  - Report artifacts
  - Nội dung Word
  - PDF export
  - Demo readiness
  - Final package

Đầu vào: bộ deliverables hiện có  
Đầu ra: checklist sẵn dùng cho nộp bài

### 4) Kết quả
- README đã có hướng dẫn chạy demo chỉ với 2 lệnh.
- Payload test API có sẵn qua `sample_request.json`.
- Checklist nộp cuối Word/PDF đã tạo và sẵn sàng tích từng mục.
- Đã xác nhận output demo thực tế:
  - `POST /predict` -> `{"fraud_probability": 0.0, "prediction": 0, "risk_level": "LOW"}`
  - `GET /health` -> `{"feature_count": 30, "model_path": "src/best_rf_model.pkl", "status": "Fraud Detection API v1.0 - Random Forest Champion"}`

### 5) File tạo/chỉnh sửa
- `README.md`
- `sample_request.json`
- `reports/final_submission_checklist.md`
