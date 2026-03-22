from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIENCE_ORDER = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
EDUCATION_ORDER = {"High School": 1, "Associate": 2, "Bachelor": 3, "Master": 4, "PhD": 5}


def parse_list_value(value: str) -> list[str]:
	if pd.isna(value):
		return []
	return [item.strip() for item in str(value).split(",") if item.strip()]


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
	for col in columns:
		if col not in df.columns:
			continue
		if not pd.api.types.is_numeric_dtype(df[col]):
			continue

		q1 = df[col].quantile(0.25)
		q3 = df[col].quantile(0.75)
		iqr = q3 - q1
		low = q1 - 1.5 * iqr
		high = q3 + 1.5 * iqr
		df[col] = df[col].clip(lower=low, upper=high)
	return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df.columns = [c.strip() for c in df.columns]

	# Harmonize salary columns across sources for easier downstream mining.
	if "salary_local" not in df.columns and "salary_usd" in df.columns:
		df["salary_local"] = df["salary_usd"]

	numeric_candidates = [
		"salary_usd",
		"salary_local",
		"years_experience",
		"remote_ratio",
		"job_description_length",
		"benefits_score",
	]
	for col in numeric_candidates:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")

	date_candidates = ["posting_date", "application_deadline"]
	for col in date_candidates:
		if col in df.columns:
			df[col] = pd.to_datetime(df[col], errors="coerce")

	if {"posting_date", "application_deadline"}.issubset(df.columns):
		df["days_to_deadline"] = (df["application_deadline"] - df["posting_date"]).dt.days

	if "required_skills" in df.columns:
		df["skills_count"] = df["required_skills"].apply(parse_list_value).apply(len)

	if {"company_location", "employee_residence"}.issubset(df.columns):
		df["home_country_match"] = (
			df["company_location"].astype(str).str.strip().str.lower()
			== df["employee_residence"].astype(str).str.strip().str.lower()
		).astype(int)

	if "experience_level" in df.columns:
		df["experience_level"] = df["experience_level"].astype(str).str.upper().str.strip()
		df["experience_level_ord"] = df["experience_level"].map(EXPERIENCE_ORDER)

	if "education_required" in df.columns:
		df["education_required"] = df["education_required"].astype(str).str.strip()
		df["education_required_ord"] = df["education_required"].map(EDUCATION_ORDER)

	text_candidates = [
		"job_title",
		"salary_currency",
		"employment_type",
		"company_location",
		"company_size",
		"employee_residence",
		"education_required",
		"industry",
		"required_skills",
		"company_name",
	]
	for col in text_candidates:
		if col in df.columns:
			df[col] = df[col].astype("string").str.strip()

	numeric_cols = df.select_dtypes(include=[np.number]).columns
	for col in numeric_cols:
		df[col] = df[col].fillna(df[col].median())

	object_cols = df.select_dtypes(include=["object", "string"]).columns
	for col in object_cols:
		mode = df[col].mode(dropna=True)
		fallback = mode.iloc[0] if not mode.empty else "Unknown"
		df[col] = df[col].fillna(fallback)

	if "job_id" in df.columns:
		df = df.drop_duplicates(subset=["job_id"], keep="first")
	else:
		df = df.drop_duplicates()

	df = cap_outliers_iqr(
		df,
		["salary_usd", "salary_local", "years_experience", "job_description_length", "benefits_score"],
	)

	for col in ["posting_date", "application_deadline"]:
		if col in df.columns:
			df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")

	return df


def summarize(df: pd.DataFrame, name: str) -> dict:
	numeric_cols = [c for c in ["salary_usd", "salary_local", "years_experience", "benefits_score"] if c in df.columns]
	num_summary = {}
	for col in numeric_cols:
		num_summary[col] = {
			"min": float(df[col].min()),
			"max": float(df[col].max()),
			"mean": float(df[col].mean()),
			"median": float(df[col].median()),
		}

	return {
		"dataset": name,
		"rows": int(df.shape[0]),
		"columns": int(df.shape[1]),
		"missing_values": int(df.isna().sum().sum()),
		"duplicate_job_id": int(df["job_id"].duplicated().sum()) if "job_id" in df.columns else None,
		"duplicate_job_id_with_source": int(df.duplicated(subset=["job_id", "source_file"]).sum())
		if {"job_id", "source_file"}.issubset(df.columns)
		else None,
		"numeric_summary": num_summary,
	}


def process_file(input_path: Path, output_path: Path, source_name: str) -> tuple[pd.DataFrame, dict]:
	raw = pd.read_csv(input_path)
	raw["source_file"] = source_name
	clean = preprocess(raw)
	clean.to_csv(output_path, index=False)
	return clean, summarize(clean, source_name)


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	file_1 = base_dir / "ai_job_dataset.csv"
	file_2 = base_dir / "ai_job_dataset1.csv"

	out_1 = base_dir / "ai_job_dataset_cleaned.csv"
	out_2 = base_dir / "ai_job_dataset1_cleaned.csv"
	out_merged = base_dir / "ai_job_dataset_merged_cleaned.csv"
	out_report = base_dir / "preprocessing_report.json"

	clean_1, report_1 = process_file(file_1, out_1, file_1.name)
	clean_2, report_2 = process_file(file_2, out_2, file_2.name)

	common_columns = sorted(set(clean_1.columns) | set(clean_2.columns))
	merged = pd.concat(
		[clean_1.reindex(columns=common_columns), clean_2.reindex(columns=common_columns)],
		ignore_index=True,
	)

	if "job_id" in merged.columns:
		merged = merged.drop_duplicates(subset=["job_id", "source_file"], keep="first")
	else:
		merged = merged.drop_duplicates()

	merged.to_csv(out_merged, index=False)
	report_merged = summarize(merged, "merged")

	report = {
		"input_files": [file_1.name, file_2.name],
		"output_files": [out_1.name, out_2.name, out_merged.name],
		"reports": [report_1, report_2, report_merged],
	}
	out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

	print("Preprocessing completed.")
	print(f"- {out_1.name}")
	print(f"- {out_2.name}")
	print(f"- {out_merged.name}")
	print(f"- {out_report.name}")


if __name__ == "__main__":
	main()
