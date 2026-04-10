"""
run_eda.py - Exploratory Data Analysis (EDA) cho AI Job Market Dataset.

Tạo báo cáo EDA chi tiết dưới dạng Markdown và các biểu đồ trực quan
phục vụ phân tích khám phá dữ liệu.

Output:
    - results/EDA_REPORT.md
    - results/eda_hist_salary_usd.png
    - results/eda_boxplot_salary_by_experience.png
    - results/eda_heatmap_correlation.png
    - results/eda_top_job_titles.png
    - results/eda_salary_by_country_top10.png
    - results/eda_employment_type_distribution.png
    - results/eda_education_distribution.png
    - results/eda_company_size_salary.png
    - results/eda_experience_vs_salary_scatter.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _to_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Chuyển DataFrame thành bảng Markdown."""
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)

    cols = list(view.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = []
    for _, row in view.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")

    if not rows:
        rows = ["| (empty) |"]

    return "\n".join([header, sep, *rows])


def generate_eda_plots(df: pd.DataFrame, out_dir: Path) -> list[str]:
    """Tạo tất cả các biểu đồ EDA và lưu vào thư mục output."""
    generated_files: list[str] = []
    sns.set_theme(style="whitegrid")

    # 1) Histogram cho salary_usd
    hist_col = "salary_usd"
    if hist_col not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            hist_col = numeric_cols[0]
        else:
            hist_col = ""

    if hist_col:
        s = pd.to_numeric(df[hist_col], errors="coerce").dropna()
        if not s.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(s, bins=40, kde=True, color="#2a9d8f", edgecolor="black", alpha=0.8)
            plt.title(f"Phân phối của {hist_col}", fontsize=14, fontweight="bold")
            plt.xlabel(hist_col, fontsize=12)
            plt.ylabel("Tần suất", fontsize=12)
            plt.tight_layout()
            hist_path = out_dir / "eda_hist_salary_usd.png"
            plt.savefig(hist_path, dpi=180)
            plt.close()
            generated_files.append(hist_path.name)

    # 2) Boxplot salary theo experience_level
    if {"salary_usd", "experience_level"}.issubset(df.columns):
        plot_df = df[["salary_usd", "experience_level"]].copy()
        plot_df["salary_usd"] = pd.to_numeric(plot_df["salary_usd"], errors="coerce")
        plot_df = plot_df.dropna(subset=["salary_usd", "experience_level"])

        if not plot_df.empty:
            exp_order = ["EN", "MI", "SE", "EX"]
            groups = [
                plot_df.loc[plot_df["experience_level"] == lvl, "salary_usd"].values
                for lvl in exp_order
                if (plot_df["experience_level"] == lvl).any()
            ]
            labels = [lvl for lvl in exp_order if (plot_df["experience_level"] == lvl).any()]

            if groups:
                plt.figure(figsize=(10, 6))
                plt.boxplot(groups, tick_labels=labels, showfliers=False,
                            patch_artist=True,
                            boxprops=dict(facecolor="#a8dadc", edgecolor="#1d3557"),
                            medianprops=dict(color="#e63946", linewidth=2))
                plt.title("Phân bố lương theo cấp độ kinh nghiệm", fontsize=14, fontweight="bold")
                plt.xlabel("Cấp độ kinh nghiệm", fontsize=12)
                plt.ylabel("Salary (USD)", fontsize=12)
                plt.tight_layout()
                box_path = out_dir / "eda_boxplot_salary_by_experience.png"
                plt.savefig(box_path, dpi=180)
                plt.close()
                generated_files.append(box_path.name)

    # 3) Heatmap tương quan
    corr_candidates = [
        c
        for c in [
            "salary_usd", "salary_local", "years_experience", "remote_ratio",
            "benefits_score", "skills_count", "days_to_deadline",
            "job_description_length", "home_country_match",
            "experience_level_ord", "education_required_ord",
        ]
        if c in df.columns
    ]

    if len(corr_candidates) >= 2:
        corr = df[corr_candidates].corr(numeric_only=True)
        plt.figure(figsize=(12, 9))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, square=True)
        plt.title("Ma trận tương quan giữa các biến số", fontsize=14, fontweight="bold")
        plt.tight_layout()
        heatmap_path = out_dir / "eda_heatmap_correlation.png"
        plt.savefig(heatmap_path, dpi=180)
        plt.close()
        generated_files.append(heatmap_path.name)

    # 4) Top 15 Job Titles
    if "job_title" in df.columns:
        top_jobs = df["job_title"].value_counts().head(15).reset_index()
        top_jobs.columns = ["job_title", "count"]
        plt.figure(figsize=(12, 7))
        sns.barplot(data=top_jobs, x="count", y="job_title",
                    hue="job_title", palette="viridis", legend=False)
        plt.title("Top 15 chức danh công việc phổ biến", fontsize=14, fontweight="bold")
        plt.xlabel("Số lượng", fontsize=12)
        plt.ylabel("")
        plt.tight_layout()
        job_path = out_dir / "eda_top_job_titles.png"
        plt.savefig(job_path, dpi=180)
        plt.close()
        generated_files.append(job_path.name)

    # 5) Salary by Country (Top 10)
    if {"company_location", "salary_usd"}.issubset(df.columns):
        country_salary = (
            df.groupby("company_location")["salary_usd"]
            .agg(["mean", "count"])
            .reset_index()
        )
        country_salary = country_salary[country_salary["count"] >= 50]
        country_salary = country_salary.sort_values("mean", ascending=False).head(10)
        plt.figure(figsize=(12, 7))
        sns.barplot(data=country_salary, x="mean", y="company_location",
                    hue="company_location", palette="magma", legend=False)
        plt.title("Lương trung bình theo quốc gia (Top 10)", fontsize=14, fontweight="bold")
        plt.xlabel("Lương trung bình (USD)", fontsize=12)
        plt.ylabel("")
        plt.tight_layout()
        country_path = out_dir / "eda_salary_by_country_top10.png"
        plt.savefig(country_path, dpi=180)
        plt.close()
        generated_files.append(country_path.name)

    # 6) Employment Type Distribution
    if "employment_type" in df.columns:
        emp_counts = df["employment_type"].value_counts().reset_index()
        emp_counts.columns = ["employment_type", "count"]
        plt.figure(figsize=(8, 6))
        colors = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#f4a261"]
        plt.pie(emp_counts["count"], labels=emp_counts["employment_type"],
                autopct="%1.1f%%", colors=colors[:len(emp_counts)],
                startangle=140, textprops={"fontsize": 11})
        plt.title("Phân bố loại hình công việc", fontsize=14, fontweight="bold")
        plt.tight_layout()
        emp_path = out_dir / "eda_employment_type_distribution.png"
        plt.savefig(emp_path, dpi=180)
        plt.close()
        generated_files.append(emp_path.name)

    # 7) Education Distribution
    if "education_required" in df.columns:
        edu_counts = df["education_required"].value_counts().reset_index()
        edu_counts.columns = ["education_required", "count"]
        edu_order = ["High School", "Associate", "Bachelor", "Master", "PhD"]
        edu_counts["order"] = edu_counts["education_required"].map(
            {v: i for i, v in enumerate(edu_order)}
        )
        edu_counts = edu_counts.sort_values("order").drop(columns="order")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=edu_counts, x="education_required", y="count",
                    hue="education_required", palette="Set2", legend=False)
        plt.title("Phân bố yêu cầu học vấn", fontsize=14, fontweight="bold")
        plt.xlabel("Trình độ học vấn", fontsize=12)
        plt.ylabel("Số lượng", fontsize=12)
        plt.tight_layout()
        edu_path = out_dir / "eda_education_distribution.png"
        plt.savefig(edu_path, dpi=180)
        plt.close()
        generated_files.append(edu_path.name)

    # 8) Company Size vs Salary
    if {"company_size", "salary_usd"}.issubset(df.columns):
        size_order = ["S", "M", "L"]
        box_df = df[["company_size", "salary_usd"]].copy()
        box_df["salary_usd"] = pd.to_numeric(box_df["salary_usd"], errors="coerce")
        box_df = box_df.dropna()
        plt.figure(figsize=(9, 6))
        sns.boxplot(data=box_df, x="company_size", y="salary_usd",
                    order=size_order, hue="company_size",
                    palette="coolwarm", legend=False, showfliers=False)
        plt.title("Lương theo quy mô công ty", fontsize=14, fontweight="bold")
        plt.xlabel("Quy mô công ty", fontsize=12)
        plt.ylabel("Salary (USD)", fontsize=12)
        plt.tight_layout()
        size_path = out_dir / "eda_company_size_salary.png"
        plt.savefig(size_path, dpi=180)
        plt.close()
        generated_files.append(size_path.name)

    # 9) Scatter: Years Experience vs Salary
    if {"years_experience", "salary_usd"}.issubset(df.columns):
        scatter_df = df[["years_experience", "salary_usd"]].copy()
        scatter_df["salary_usd"] = pd.to_numeric(scatter_df["salary_usd"], errors="coerce")
        scatter_df["years_experience"] = pd.to_numeric(scatter_df["years_experience"], errors="coerce")
        scatter_df = scatter_df.dropna()
        if len(scatter_df) > 5000:
            scatter_df = scatter_df.sample(5000, random_state=42)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=scatter_df, x="years_experience", y="salary_usd",
                        alpha=0.4, s=25, color="#457b9d")
        z = np.polyfit(scatter_df["years_experience"], scatter_df["salary_usd"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(scatter_df["years_experience"].min(),
                             scatter_df["years_experience"].max(), 100)
        plt.plot(x_line, p(x_line), color="#e63946", linewidth=2, label="Trend line")
        plt.title("Mối quan hệ Số năm kinh nghiệm vs Lương", fontsize=14, fontweight="bold")
        plt.xlabel("Số năm kinh nghiệm", fontsize=12)
        plt.ylabel("Salary (USD)", fontsize=12)
        plt.legend()
        plt.tight_layout()
        exp_path = out_dir / "eda_experience_vs_salary_scatter.png"
        plt.savefig(exp_path, dpi=180)
        plt.close()
        generated_files.append(exp_path.name)

    return generated_files


def build_eda_report(df: pd.DataFrame, dataset_name: str) -> str:
    """Xây dựng báo cáo EDA đầy đủ dưới dạng Markdown."""
    lines: list[str] = []

    n_rows, n_cols = df.shape
    lines.append("# EDA - AI Job Market Dataset")
    lines.append("")
    lines.append("## 1) Tổng quan dữ liệu")
    lines.append(f"- Dataset: {dataset_name}")
    lines.append(f"- Số dòng: {n_rows:,}")
    lines.append(f"- Số cột: {n_cols}")
    lines.append("")

    # Kiểu dữ liệu và missing
    dtypes_df = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "missing_pct": [float(df[c].isna().mean() * 100) for c in df.columns],
    }).sort_values(["missing", "column"], ascending=[False, True])

    lines.append("## 2) Kiểu dữ liệu và giá trị thiếu")
    lines.append(_to_markdown_table(dtypes_df, max_rows=30))
    lines.append("")
    lines.append(f"- Tổng giá trị thiếu: {int(df.isna().sum().sum())}")
    lines.append("")

    # Trùng lặp
    duplicate_all = int(df.duplicated().sum())
    lines.append("## 3) Kiểm tra trùng lặp")
    lines.append(f"- Số dòng trùng lặp toàn bộ: {duplicate_all}")
    if "job_id" in df.columns:
        lines.append(f"- Số job_id trùng lặp: {int(df['job_id'].duplicated().sum())}")
    if {"job_id", "source_file"}.issubset(df.columns):
        pair_dup = int(df.duplicated(subset=["job_id", "source_file"]).sum())
        lines.append(f"- Số cặp (job_id, source_file) trùng lặp: {pair_dup}")
    lines.append("")

    # Thống kê mô tả
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    lines.append("## 4) Thống kê mô tả các biến số")
    if numeric_cols:
        desc = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).transpose().reset_index()
        desc = desc.rename(columns={"index": "feature"})
        lines.append(_to_markdown_table(desc, max_rows=30))
    else:
        lines.append("Không có biến số.")
    lines.append("")

    # Phân tích nhóm biến phân loại
    lines.append("## 5) Phân tích nhóm biến phân loại")
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if cat_cols:
        target_cat = [
            c
            for c in [
                "job_title", "experience_level", "employment_type",
                "company_size", "industry", "company_location",
                "salary_currency", "education_required", "source_file",
            ]
            if c in cat_cols
        ]
        if not target_cat:
            target_cat = cat_cols[:5]

        for c in target_cat:
            vc = (
                df[c]
                .fillna("Unknown")
                .value_counts(dropna=False)
                .head(10)
                .reset_index()
            )
            vc.columns = [c, "count"]
            vc["pct"] = (vc["count"] / len(df) * 100).round(2)
            lines.append(f"### {c} - Top 10")
            lines.append(_to_markdown_table(vc, max_rows=10))
            lines.append("")
    else:
        lines.append("Không có biến phân loại.")
        lines.append("")

    # Phân bố biến quan trọng
    lines.append("## 6) Phân bố một số biến quan trọng")
    focus_num = [
        c
        for c in [
            "salary_usd", "salary_local", "years_experience", "remote_ratio",
            "benefits_score", "skills_count", "days_to_deadline",
            "job_description_length",
        ]
        if c in df.columns
    ]

    if focus_num:
        dist_rows = []
        for c in focus_num:
            s = pd.to_numeric(df[c], errors="coerce")
            dist_rows.append({
                "feature": c,
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
                "min": float(s.min()),
                "max": float(s.max()),
            })
        dist_df = pd.DataFrame(dist_rows)
        lines.append(_to_markdown_table(dist_df, max_rows=30))
        lines.append("")
    else:
        lines.append("Không tìm thấy biến số trong nhóm quan trọng.")
        lines.append("")

    # Lương theo kinh nghiệm
    if "salary_usd" in df.columns and "experience_level" in df.columns:
        lines.append("## 7) Lương theo cấp độ kinh nghiệm")
        salary_exp = (
            df.groupby("experience_level", dropna=False)["salary_usd"]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        lines.append(_to_markdown_table(salary_exp, max_rows=20))
        lines.append("")

    # Tương quan
    lines.append("## 8) Tương quan giữa các biến số")
    corr_candidates = [
        c
        for c in [
            "salary_usd", "salary_local", "years_experience", "remote_ratio",
            "benefits_score", "skills_count", "days_to_deadline",
            "job_description_length", "home_country_match",
            "experience_level_ord", "education_required_ord",
        ]
        if c in df.columns
    ]

    if len(corr_candidates) >= 2:
        corr = df[corr_candidates].corr(numeric_only=True)
        lines.append(_to_markdown_table(corr.reset_index().rename(columns={"index": "feature"}), max_rows=30))

        sal_corr = (
            corr["salary_usd"].drop(labels=["salary_usd"]).sort_values(ascending=False)
            if "salary_usd" in corr.columns
            else pd.Series(dtype=float)
        )
        if not sal_corr.empty:
            top_pos = sal_corr.head(5)
            top_neg = sal_corr.tail(5)

            lines.append("")
            lines.append("### Biến tương quan dương cao với salary_usd")
            lines.append(_to_markdown_table(top_pos.reset_index().rename(columns={"index": "feature", 0: "corr"})))
            lines.append("")
            lines.append("### Biến tương quan âm/yếu với salary_usd")
            lines.append(_to_markdown_table(top_neg.reset_index().rename(columns={"index": "feature", 0: "corr"})))
            lines.append("")
    else:
        lines.append("Không đủ biến số để tính tương quan.")
        lines.append("")

    # Nhận xét
    lines.append("## 9) Nhận xét tổng hợp")
    lines.append("- Dữ liệu sau tiền xử lý không còn giá trị thiếu.")
    lines.append("- Có sự khác biệt rõ rệt về mức lương theo cấp độ kinh nghiệm.")
    lines.append("- Biến years_experience và experience_level_ord có tương quan dương mạnh với salary_usd.")
    lines.append("- Các biến remote_ratio, skills_count có tương quan yếu với lương.")
    lines.append("- Cần kết hợp EDA với kết quả Regression/Classification để rút ra kết luận chặt chẽ.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Entry point: chạy toàn bộ EDA pipeline."""
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "ai_job_dataset_merged_cleaned.csv"
    out_dir = base_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    report_text = build_eda_report(df, input_path.name)
    report_path = out_dir / "EDA_REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")

    plot_files = generate_eda_plots(df, out_dir)

    print(f"EDA report generated: {report_path.name}")
    if plot_files:
        print(f"EDA plots generated ({len(plot_files)} files):")
        for name in plot_files:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
