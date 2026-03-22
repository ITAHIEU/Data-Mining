from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
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
    generated_files: list[str] = []

    # 1) Histogram for salary_usd (or first numeric column as fallback).
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
            plt.hist(s, bins=40, edgecolor="black", alpha=0.8)
            plt.title(f"Histogram of {hist_col}")
            plt.xlabel(hist_col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            hist_path = out_dir / "eda_hist_salary_usd.png"
            plt.savefig(hist_path, dpi=180)
            plt.close()
            generated_files.append(hist_path.name)

    # 2) Boxplot for salary_usd by experience_level when available.
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
                plt.boxplot(groups, tick_labels=labels, showfliers=False)
                plt.title("Salary USD by Experience Level")
                plt.xlabel("Experience Level")
                plt.ylabel("Salary USD")
                plt.tight_layout()
                box_path = out_dir / "eda_boxplot_salary_by_experience.png"
                plt.savefig(box_path, dpi=180)
                plt.close()
                generated_files.append(box_path.name)

    # 3) Heatmap for numeric correlation matrix.
    corr_candidates = [
        c
        for c in [
            "salary_usd",
            "salary_local",
            "years_experience",
            "remote_ratio",
            "benefits_score",
            "skills_count",
            "days_to_deadline",
            "job_description_length",
            "home_country_match",
            "experience_level_ord",
            "education_required_ord",
        ]
        if c in df.columns
    ]

    if len(corr_candidates) >= 2:
        corr = df[corr_candidates].corr(numeric_only=True)
        plt.figure(figsize=(11, 8))
        im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation Heatmap")
        cbar = plt.colorbar(im)
        cbar.set_label("Correlation")
        plt.tight_layout()
        heatmap_path = out_dir / "eda_heatmap_correlation.png"
        plt.savefig(heatmap_path, dpi=180)
        plt.close()
        generated_files.append(heatmap_path.name)

    return generated_files


def build_eda_report(df: pd.DataFrame, dataset_name: str) -> str:
    lines: list[str] = []

    n_rows, n_cols = df.shape
    lines.append("# EDA - AI Job Dataset")
    lines.append("")
    lines.append("## 1) Tong quan du lieu")
    lines.append(f"- Dataset: {dataset_name}")
    lines.append(f"- So dong: {n_rows}")
    lines.append(f"- So cot: {n_cols}")
    lines.append("")

    dtypes_df = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "missing_pct": [float(df[c].isna().mean() * 100) for c in df.columns],
    }).sort_values(["missing", "column"], ascending=[False, True])

    lines.append("## 2) Kieu du lieu va thieu du lieu")
    lines.append(_to_markdown_table(dtypes_df, max_rows=30))
    lines.append("")
    lines.append(f"- Tong gia tri thieu: {int(df.isna().sum().sum())}")
    lines.append("")

    duplicate_all = int(df.duplicated().sum())
    lines.append("## 3) Kiem tra trung lap")
    lines.append(f"- So dong trung lap toan bo: {duplicate_all}")
    if "job_id" in df.columns:
        lines.append(f"- So job_id trung lap: {int(df['job_id'].duplicated().sum())}")
    if {"job_id", "source_file"}.issubset(df.columns):
        pair_dup = int(df.duplicated(subset=["job_id", "source_file"]).sum())
        lines.append(f"- So cap (job_id, source_file) trung lap: {pair_dup}")
    lines.append("")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    lines.append("## 4) Thong ke mo ta bien so")
    if numeric_cols:
        desc = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).transpose().reset_index()
        desc = desc.rename(columns={"index": "feature"})
        lines.append(_to_markdown_table(desc, max_rows=30))
    else:
        lines.append("Khong co bien so.")
    lines.append("")

    lines.append("## 5) Phan tich nhom bien phan loai")
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if cat_cols:
        target_cat = [
            c
            for c in [
                "job_title",
                "experience_level",
                "employment_type",
                "company_size",
                "industry",
                "company_location",
                "salary_currency",
                "source_file",
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
        lines.append("Khong co bien phan loai.")
        lines.append("")

    lines.append("## 6) Phan bo mot so bien quan trong")
    focus_num = [
        c
        for c in [
            "salary_usd",
            "salary_local",
            "years_experience",
            "remote_ratio",
            "benefits_score",
            "skills_count",
            "days_to_deadline",
            "job_description_length",
        ]
        if c in df.columns
    ]

    if focus_num:
        dist_rows = []
        for c in focus_num:
            s = pd.to_numeric(df[c], errors="coerce")
            dist_rows.append(
                {
                    "feature": c,
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std()),
                    "p25": float(s.quantile(0.25)),
                    "p75": float(s.quantile(0.75)),
                    "min": float(s.min()),
                    "max": float(s.max()),
                }
            )
        dist_df = pd.DataFrame(dist_rows)
        lines.append(_to_markdown_table(dist_df, max_rows=30))
        lines.append("")
    else:
        lines.append("Khong tim thay bien so trong nhom quan trong.")
        lines.append("")

    if "salary_usd" in df.columns and "experience_level" in df.columns:
        lines.append("## 7) Luong theo cap do kinh nghiem")
        salary_exp = (
            df.groupby("experience_level", dropna=False)["salary_usd"]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        lines.append(_to_markdown_table(salary_exp, max_rows=20))
        lines.append("")

    lines.append("## 8) Tuong quan giua cac bien so")
    corr_candidates = [
        c
        for c in [
            "salary_usd",
            "salary_local",
            "years_experience",
            "remote_ratio",
            "benefits_score",
            "skills_count",
            "days_to_deadline",
            "job_description_length",
            "home_country_match",
            "experience_level_ord",
            "education_required_ord",
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
            lines.append("### Bien tuong quan duong cao voi salary_usd")
            lines.append(_to_markdown_table(top_pos.reset_index().rename(columns={"index": "feature", 0: "corr"})))
            lines.append("")
            lines.append("### Bien tuong quan am cao voi salary_usd")
            lines.append(_to_markdown_table(top_neg.reset_index().rename(columns={"index": "feature", 0: "corr"})))
            lines.append("")
    else:
        lines.append("Khong du bien so de tinh tuong quan.")
        lines.append("")

    lines.append("## 9) Nhan xet nhanh")
    lines.append("- Du lieu sau tien xu ly khong con gia tri thieu.")
    lines.append("- Co su khac biet ve muc luong theo cap do kinh nghiem va cac dac trung lien quan.")
    lines.append("- Nen ket hop EDA voi ket qua Regression/Classification de ket luan chat che hon.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
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
        print("EDA plots generated:")
        for name in plot_files:
            print(f"- {name}")


if __name__ == "__main__":
    main()
