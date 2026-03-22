# Data-Mining

Du an Data Mining phan tich thi truong viec lam AI, gom cac bai toan:

- Tien xu ly va lam sach du lieu
- EDA (kham pha du lieu)
- Regression (du doan salary_usd)
- Classification (du doan high salary)
- Clustering (phan nhom viec lam)
- Association Rules (luat ket hop ky nang)

## 1) Cau truc du an

- Process.py: lam sach du lieu, merge 2 file csv, sinh preprocessing_report.json
- run_eda.py: tao bao cao EDA va cac hinh EDA
- run_topic_analysis.py: chay regression, classification, clustering, association rules va tao toan bo ket qua trong thu muc results
- results/: thu muc ket qua csv, markdown report, va hinh anh

## 2) Yeu cau moi truong

- Python 3.10+ (khuyen nghi 3.10 hoac 3.11)
- PowerShell (Windows)

Thu vien Python can dung:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- mlxtend

## 3) Cai dat va chay chi tiet (Windows - PowerShell)

### Buoc 1: Mo terminal tai thu muc du an

Vi du duong dan:

E:/Tai lieu/data mining

### Buoc 2: Tao va kich hoat virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Neu gap loi execution policy, chay tam thoi:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Buoc 3: Cai thu vien

```powershell
python -m pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend
```

### Buoc 4: Chay tien xu ly du lieu

```powershell
python Process.py
```

Output du kien sinh ra o thu muc goc:

- ai_job_dataset_cleaned.csv
- ai_job_dataset1_cleaned.csv
- ai_job_dataset_merged_cleaned.csv
- preprocessing_report.json

### Buoc 5: Chay EDA

```powershell
python run_eda.py
```

Output du kien trong results:

- EDA_REPORT.md
- eda_hist_salary_usd.png
- eda_boxplot_salary_by_experience.png
- eda_heatmap_correlation.png

### Buoc 6: Chay phan tich mo hinh tong hop

```powershell
python run_topic_analysis.py
```

Output du kien trong results:

- regression_results.csv
- classification_results.csv
- clustering_scores.csv
- cluster_profile.csv
- association_rules_top20.csv
- PROJECT_REPORT.md
- regression_comparison.png
- classification_comparison.png
- cluster_scatter_pca.png
- top_skill_support.png
- detail_salary_distribution.png
- detail_salary_by_experience_boxplot.png
- detail_remote_ratio_vs_salary.png
- detail_regression_actual_vs_pred.png
- detail_regression_residual_distribution.png
- detail_classification_confusion_matrix.png
- detail_classification_roc_curves.png
- detail_clustering_silhouette_by_k.png
- detail_cluster_size_distribution.png
- detail_association_support_confidence_lift.png

## 4) Thu tu chay khuyen nghi

Chay dung thu tu nay de tranh loi file dau vao:

1. python Process.py
2. python run_eda.py
3. python run_topic_analysis.py

Ly do: run_topic_analysis.py can file ai_job_dataset_merged_cleaned.csv do Process.py tao ra truoc.

## 5) Kiem tra nhanh sau khi chay

- Kiem tra file ai_job_dataset_merged_cleaned.csv da ton tai
- Kiem tra file results/PROJECT_REPORT.md da duoc tao
- Kiem tra cac file csv ket qua trong results

## 6) Day len GitHub (neu can)

Lenh day ma nguon va ket qua len repo:

```powershell
git add .
git commit -m "Update README with detailed run guide"
git push
```

## 7) Loi thuong gap va cach xu ly

### Loi: Missing ai_job_dataset_merged_cleaned.csv

Nguyen nhan: chua chay Process.py.

Cach xu ly:

```powershell
python Process.py
python run_topic_analysis.py
```

### Loi: ModuleNotFoundError

Nguyen nhan: chua cai du thu vien trong virtual environment dang dung.

Cach xu ly:

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend
```

### Loi: Khong kich hoat duoc .venv

Cach xu ly:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
