# Huong Dan Deploy Web

## 1) Chay local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Mac dinh web se mo tai:

- http://localhost:8501

## 2) Deploy len Streamlit Community Cloud

1. Push nhanh `feature/web-job-advisor` len GitHub.
2. Vao Streamlit Community Cloud.
3. Chon repository: `ITAHIEU/Data-Mining`.
4. Branch: `feature/web-job-advisor` (hoac merge vao `main` roi chon `main`).
5. Main file path: `streamlit_app.py`.
6. Deploy.

## 3) Luu y du lieu

- Web dung truc tiep file `ai_job_dataset_merged_cleaned.csv` trong repo.
- Neu thay doi du lieu, can push lai file csv de web cap nhat.
