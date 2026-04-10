# Data Mining - Phân tích thị trường việc làm AI 2025

## Đề tài
**Phân tích thị trường việc làm và xu hướng lương ngành AI trong năm 2025**

Dự án áp dụng các kỹ thuật Data Mining để phân tích bộ dữ liệu "Global AI Job Market & Salary Trends 2025" (30,000+ bản ghi), bao gồm:

- **Tiền xử lý dữ liệu** (Data Preprocessing): Làm sạch, xử lý missing values, outliers, feature engineering
- **Phân tích khám phá dữ liệu** (EDA): Thống kê mô tả, trực quan hóa, phân tích tương quan
- **Hồi quy** (Regression): Dự đoán mức lương (salary_usd) - 5 mô hình
- **Phân loại** (Classification): Dự đoán nhóm lương cao - 5 mô hình
- **Phân cụm** (Clustering): Phân nhóm công việc - 3 thuật toán (KMeans, Agglomerative, DBSCAN)
- **Luật kết hợp** (Association Rules): Khám phá kỹ năng thường đi cùng nhau
- **Cross-Validation**: 5-fold CV cho tất cả mô hình
- **Feature Importance**: Phân tích đặc trưng quan trọng
- **Learning Curves**: Đánh giá overfitting/underfitting
- **Hyperparameter Tuning**: GridSearchCV tối ưu siêu tham số

## Thông tin nhóm

| Thành viên | MSSV |
|-----------|------|
| Nguyễn Lê Đức Hiếu | 2210997 |
| Nguyễn Thị Thúy Nga | 2212168 |
| Nguyễn Huỳnh Thái Bảo | 2210238 |

**GVHD**: Th.S Đỗ Thành Thái

## Cấu trúc dự án

```
Data-Mining/
├── main.py                  # Entry point - chạy toàn bộ pipeline
├── Process.py               # Tiền xử lý, merge dữ liệu
├── run_eda.py               # Phân tích khám phá (EDA) + biểu đồ
├── run_topic_analysis.py    # Regression, Classification, Clustering,
│                            # Association Rules, Feature Importance,
│                            # Learning Curves, GridSearchCV
├── requirements.txt         # Thư viện Python cần thiết
├── ai_job_dataset.csv       # Dữ liệu gốc 1
├── ai_job_dataset1.csv      # Dữ liệu gốc 2
├── MODEL_DETAILS.md         # Chi tiết mô hình
├── README.md                # File này
└── results/                 # Thư mục kết quả
    ├── regression_results.csv
    ├── classification_results.csv
    ├── clustering_scores.csv
    ├── cluster_profile.csv
    ├── dbscan_results.csv
    ├── association_rules_top20.csv
    ├── feature_importance_regression.csv
    ├── feature_importance_classification.csv
    ├── gridsearch_results.csv
    ├── gridsearch_best_params.csv
    ├── PROJECT_REPORT.md
    ├── EDA_REPORT.md
    └── *.png                # 25+ biểu đồ
```

## Yêu cầu môi trường

- Python 3.10+ (khuyến nghị 3.11 hoặc 3.12)
- Windows / macOS / Linux

## Cài đặt và chạy

### Bước 1: Clone repository

```bash
git clone https://github.com/ITAHIEU/Data-Mining.git
cd Data-Mining
```

### Bước 2: Tạo virtual environment

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Nếu gặp lỗi execution policy:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Bước 3: Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 4: Chạy toàn bộ pipeline

```bash
python main.py
```

Hoặc chạy từng bước:

```bash
# Bước 1: Tiền xử lý
python Process.py

# Bước 2: EDA
python run_eda.py

# Bước 3: Phân tích mô hình
python run_topic_analysis.py
```

## Mô hình sử dụng

### Hồi quy (Regression) - Dự đoán salary_usd

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Random Forest | 21,820 | 16,407 | 0.868 |
| Gradient Boosting | ~22,000 | ~16,500 | ~0.865 |
| XGBoost | ~21,900 | ~16,400 | ~0.867 |
| Linear Regression | 23,374 | 17,906 | 0.849 |
| Decision Tree | 23,789 | 17,430 | 0.843 |

### Phân loại (Classification) - Dự đoán high salary

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Random Forest | 0.927 | 0.853 | 0.979 |
| Gradient Boosting | ~0.928 | ~0.854 | ~0.980 |
| XGBoost | ~0.929 | ~0.855 | ~0.981 |
| Logistic Regression | 0.927 | 0.852 | 0.980 |
| Decision Tree | 0.920 | 0.840 | 0.959 |

### Phân cụm (Clustering)

- **KMeans** (best): k=2, silhouette=0.225
- **Agglomerative**: k=2, silhouette=0.215
- **DBSCAN**: Density-based, không cần chọn k

### Luật kết hợp (Association Rules)

- TensorFlow → Python: confidence=0.426, lift=1.429
- Python → TensorFlow: confidence=0.291, lift=1.429

## Nguồn dữ liệu

- [Global AI Job Market & Salary Trends 2025 (Kaggle)](https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025)

## Triển khai (Deploy)

- [Web App (Streamlit)](https://job-ai-datamining.streamlit.app/)

## Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|------------|------------|
| Missing CSV | Chưa chạy Process.py | `python Process.py` |
| ModuleNotFoundError | Thiếu thư viện | `pip install -r requirements.txt` |
| XGBoost warning | Chưa cài xgboost | `pip install xgboost` (tùy chọn) |
