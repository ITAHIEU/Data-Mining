from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


RANDOM_STATE = 42


def ensure_columns(df: pd.DataFrame, columns: list[str], fill_value: float = 0.0) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df


def parse_skills(series: pd.Series) -> list[list[str]]:
    transactions = []
    for value in series.fillna(""):
        items = [x.strip() for x in str(value).split(",") if x.strip()]
        transactions.append(items)
    return transactions


def build_feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric_cols = [
        "years_experience",
        "remote_ratio",
        "job_description_length",
        "benefits_score",
        "skills_count",
        "days_to_deadline",
        "home_country_match",
        "experience_level_ord",
        "education_required_ord",
    ]
    categorical_cols = [
        "experience_level",
        "employment_type",
        "company_size",
        "industry",
        "company_location",
        "salary_currency",
    ]

    df = ensure_columns(df, numeric_cols, fill_value=0.0)
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "Unknown"

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].astype("string").fillna("Unknown")

    features = df[numeric_cols + categorical_cols].copy()
    return features, numeric_cols, categorical_cols


def regression_experiment(
    df: pd.DataFrame,
    features: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, str, dict[str, dict[str, np.ndarray]]]:
    y = pd.to_numeric(df["salary_usd"], errors="coerce")
    valid_mask = y.notna()
    X = features.loc[valid_mask]
    y = y.loc[valid_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    models: dict[str, Pipeline] = {
        "LinearRegression": Pipeline(
            [("preprocessor", preprocessor), ("model", LinearRegression())]
        ),
        "DecisionTreeRegressor": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
    }

    rows = []
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
        rows.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
        diagnostics[name] = {
            "y_true": y_test.to_numpy(),
            "y_pred": pred,
        }

    results = pd.DataFrame(rows).sort_values(by="rmse", ascending=True).reset_index(drop=True)
    best_model = str(results.iloc[0]["model"])
    return results, best_model, diagnostics


def classification_experiment(
    df: pd.DataFrame,
    features: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, str, dict[str, dict[str, np.ndarray]]]:
    salary = pd.to_numeric(df["salary_usd"], errors="coerce")
    threshold = float(salary.quantile(0.75))
    target = (salary >= threshold).astype(int)

    valid_mask = salary.notna()
    X = features.loc[valid_mask]
    y = target.loc[valid_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    models = {
        "LogisticRegression": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "DecisionTreeClassifier": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
    }

    rows = []
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        if hasattr(model.named_steps["model"], "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, prob))
        else:
            prob = pred.astype(float)
            auc = np.nan

        rows.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(y_test, pred)),
                "f1": float(f1_score(y_test, pred)),
                "roc_auc": auc,
                "salary_threshold": threshold,
            }
        )

        diagnostics[name] = {
            "y_true": y_test.to_numpy(),
            "y_pred": pred,
            "y_prob": prob,
        }

    results = pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)
    best_model = str(results.iloc[0]["model"])
    return results, best_model, diagnostics


def clustering_experiment(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    cluster_features = [
        "salary_usd",
        "years_experience",
        "remote_ratio",
        "benefits_score",
        "skills_count",
        "job_description_length",
    ]
    df = ensure_columns(df.copy(), cluster_features, fill_value=0.0)

    X = df[cluster_features].copy()
    for col in cluster_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sample_size = min(6000, len(X))
    if sample_size < len(X):
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(len(X), size=sample_size, replace=False)
        X_eval = X_scaled[sample_idx]
    else:
        X_eval = X_scaled

    rows = []
    best_k = 2
    best_score = -1.0

    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels_kmeans = kmeans.fit_predict(X_eval)
        score_kmeans = float(silhouette_score(X_eval, labels_kmeans))

        agg = AgglomerativeClustering(n_clusters=k)
        labels_agg = agg.fit_predict(X_eval)
        score_agg = float(silhouette_score(X_eval, labels_agg))

        rows.append(
            {
                "k": k,
                "kmeans_silhouette": score_kmeans,
                "agglomerative_silhouette": score_agg,
            }
        )

        if score_kmeans > best_score:
            best_score = score_kmeans
            best_k = k

    clustering_scores = pd.DataFrame(rows)

    best_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    df["cluster"] = best_kmeans.fit_predict(X_scaled)

    cluster_profile = (
        df.groupby("cluster")[cluster_features]
        .mean()
        .round(2)
        .reset_index()
        .sort_values(by="cluster")
    )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_data = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"pc1": pca_data[:, 0], "pc2": pca_data[:, 1], "cluster": df["cluster"]})

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="pc1", y="pc2", hue="cluster", palette="tab10", s=30)
    plt.title(f"KMeans Clusters (k={best_k}) in PCA Space")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_scatter_pca.png", dpi=140)
    plt.close()

    cluster_profile.to_csv(output_dir / "cluster_profile.csv", index=False)
    clustering_scores.to_csv(output_dir / "clustering_scores.csv", index=False)

    return clustering_scores, best_k, df


def association_experiment(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = parse_skills(df["required_skills"])
    transactions = [t for t in transactions if len(t) > 0]

    te = TransactionEncoder()
    te_arr = te.fit(transactions).transform(transactions)
    skill_df = pd.DataFrame(te_arr, columns=te.columns_)

    support = skill_df.mean().sort_values(ascending=False).reset_index()
    support.columns = ["skill", "support"]
    top_support = support.head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_support,
        x="support",
        y="skill",
        hue="skill",
        palette="viridis",
        legend=False,
    )
    plt.title("Top 15 Skill Support")
    plt.tight_layout()
    plt.savefig(output_dir / "top_skill_support.png", dpi=140)
    plt.close()

    frequent = apriori(skill_df, min_support=0.03, use_colnames=True)
    frequent = frequent.sort_values(by="support", ascending=False)

    if frequent.empty:
        rules = pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )
    else:
        rules = association_rules(frequent, metric="lift", min_threshold=1.1)
        if not rules.empty:
            rules = rules.sort_values(by=["lift", "confidence"], ascending=False)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
            rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
            rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]

    top_rules = rules.head(20)
    top_rules.to_csv(output_dir / "association_rules_top20.csv", index=False)

    return top_rules, top_support


def save_detailed_visualizations(
    df: pd.DataFrame,
    output_dir: Path,
    reg_results: pd.DataFrame,
    cls_results: pd.DataFrame,
    clustering_scores: pd.DataFrame,
    clustered_df: pd.DataFrame,
    reg_diagnostics: dict[str, dict[str, np.ndarray]],
    cls_diagnostics: dict[str, dict[str, np.ndarray]],
    assoc_rules: pd.DataFrame,
) -> None:
    salary = pd.to_numeric(df["salary_usd"], errors="coerce").dropna()

    plt.figure(figsize=(9, 5))
    sns.histplot(salary, bins=40, kde=True, color="#2a9d8f")
    plt.title("Salary Distribution (USD)")
    plt.xlabel("salary_usd")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_salary_distribution.png", dpi=140)
    plt.close()

    if "experience_level" in df.columns:
        box_df = df[["experience_level", "salary_usd"]].copy()
        box_df["salary_usd"] = pd.to_numeric(box_df["salary_usd"], errors="coerce")
        box_df = box_df.dropna()
        order = ["EN", "MI", "SE", "EX"]
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=box_df,
            x="experience_level",
            y="salary_usd",
            order=order,
            hue="experience_level",
            palette="Set2",
            legend=False,
        )
        plt.title("Salary by Experience Level")
        plt.tight_layout()
        plt.savefig(output_dir / "detail_salary_by_experience_boxplot.png", dpi=140)
        plt.close()

    scatter_df = df[["remote_ratio", "salary_usd", "experience_level"]].copy()
    scatter_df["salary_usd"] = pd.to_numeric(scatter_df["salary_usd"], errors="coerce")
    scatter_df["remote_ratio"] = pd.to_numeric(scatter_df["remote_ratio"], errors="coerce")
    scatter_df = scatter_df.dropna()
    if len(scatter_df) > 5000:
        scatter_df = scatter_df.sample(5000, random_state=RANDOM_STATE)
    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=scatter_df,
        x="remote_ratio",
        y="salary_usd",
        hue="experience_level",
        alpha=0.6,
        s=28,
        palette="tab10",
    )
    plt.title("Remote Ratio vs Salary")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_remote_ratio_vs_salary.png", dpi=140)
    plt.close()

    best_reg_model = str(reg_results.iloc[0]["model"])
    best_reg_diag = reg_diagnostics[best_reg_model]
    y_true_reg = best_reg_diag["y_true"]
    y_pred_reg = best_reg_diag["y_pred"]

    reg_min = float(min(np.min(y_true_reg), np.min(y_pred_reg)))
    reg_max = float(max(np.max(y_true_reg), np.max(y_pred_reg)))
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true_reg, y=y_pred_reg, alpha=0.35, s=20, color="#1d3557")
    plt.plot([reg_min, reg_max], [reg_min, reg_max], color="#e63946", linewidth=2)
    plt.title(f"Actual vs Predicted Salary ({best_reg_model})")
    plt.xlabel("Actual salary_usd")
    plt.ylabel("Predicted salary_usd")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_regression_actual_vs_pred.png", dpi=140)
    plt.close()

    residual = y_true_reg - y_pred_reg
    plt.figure(figsize=(8, 5))
    sns.histplot(residual, bins=40, kde=True, color="#6a4c93")
    plt.title(f"Residual Distribution ({best_reg_model})")
    plt.xlabel("Residual = y_true - y_pred")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_regression_residual_distribution.png", dpi=140)
    plt.close()

    best_cls_model = str(cls_results.iloc[0]["model"])
    best_cls_diag = cls_diagnostics[best_cls_model]
    y_true_cls = best_cls_diag["y_true"]
    y_pred_cls = best_cls_diag["y_pred"]

    cm = confusion_matrix(y_true_cls, y_pred_cls)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix ({best_cls_model})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_classification_confusion_matrix.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 6))
    for model_name, diag in cls_diagnostics.items():
        fpr, tpr, _ = roc_curve(diag["y_true"], diag["y_prob"])
        auc_value = roc_auc_score(diag["y_true"], diag["y_prob"])
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curves - Classification Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_classification_roc_curves.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    melt_cluster = clustering_scores.melt(
        id_vars="k",
        value_vars=["kmeans_silhouette", "agglomerative_silhouette"],
        var_name="algorithm",
        value_name="silhouette",
    )
    sns.lineplot(data=melt_cluster, x="k", y="silhouette", hue="algorithm", marker="o")
    plt.title("Silhouette Score by k")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_clustering_silhouette_by_k.png", dpi=140)
    plt.close()

    cluster_sizes = clustered_df["cluster"].value_counts().sort_index().reset_index()
    cluster_sizes.columns = ["cluster", "count"]
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=cluster_sizes, x="cluster", y="count", hue="cluster", palette="tab10", legend=False)
    plt.title("Cluster Size Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_cluster_size_distribution.png", dpi=140)
    plt.close()

    if not assoc_rules.empty:
        plot_rules = assoc_rules.head(15).copy()
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=plot_rules,
            x="support",
            y="confidence",
            size="lift",
            hue="lift",
            sizes=(50, 350),
            palette="viridis",
        )
        plt.title("Association Rules: Support vs Confidence")
        plt.tight_layout()
        plt.savefig(output_dir / "detail_association_support_confidence_lift.png", dpi=140)
        plt.close()


def save_model_plots(reg_results: pd.DataFrame, cls_results: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=reg_results,
        x="rmse",
        y="model",
        hue="model",
        palette="magma",
        legend=False,
    )
    plt.title("Regression Model Comparison (Lower RMSE Better)")
    plt.tight_layout()
    plt.savefig(output_dir / "regression_comparison.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=cls_results,
        x="f1",
        y="model",
        hue="model",
        palette="crest",
        legend=False,
    )
    plt.title("Classification Model Comparison (Higher F1 Better)")
    plt.tight_layout()
    plt.savefig(output_dir / "classification_comparison.png", dpi=140)
    plt.close()


def write_report(
    output_dir: Path,
    reg_results: pd.DataFrame,
    cls_results: pd.DataFrame,
    clustering_scores: pd.DataFrame,
    best_k: int,
    assoc_rules: pd.DataFrame,
) -> None:
    best_reg = reg_results.iloc[0]
    best_cls = cls_results.iloc[0]
    best_cluster_row = clustering_scores.loc[clustering_scores["k"] == best_k].iloc[0]

    lines = []
    lines.append("# Data Mining Project Report")
    lines.append("")
    lines.append("## Topic")
    lines.append(
        "AI Job Market Intelligence: salary prediction, high-salary classification, job clustering, and skill association rules."
    )
    lines.append("")
    lines.append("## Methods")
    lines.append("- Regression: Linear Regression vs Decision Tree Regressor vs Random Forest Regressor")
    lines.append("- Classification: Logistic Regression vs Decision Tree Classifier vs Random Forest Classifier")
    lines.append("- Clustering: KMeans vs Agglomerative Clustering (evaluated by silhouette score)")
    lines.append("- Association Rule Mining: Apriori + Lift/Confidence")
    lines.append("")
    lines.append("## Key Results")
    lines.append(
        f"- Best regression model: {best_reg['model']} (RMSE={best_reg['rmse']:.2f}, MAE={best_reg['mae']:.2f}, R2={best_reg['r2']:.4f})"
    )
    lines.append(
        f"- Best classification model: {best_cls['model']} (F1={best_cls['f1']:.4f}, Accuracy={best_cls['accuracy']:.4f}, ROC_AUC={best_cls['roc_auc']:.4f})"
    )
    lines.append(
        f"- Best KMeans setup: k={best_k}, silhouette={best_cluster_row['kmeans_silhouette']:.4f}"
    )

    if assoc_rules.empty:
        lines.append("- Association rules: no rule satisfied support/lift threshold.")
    else:
        top_rule = assoc_rules.iloc[0]
        lines.append(
            "- Strongest association rule: "
            + f"{top_rule['antecedents']} -> {top_rule['consequents']} "
            + f"(support={top_rule['support']:.4f}, confidence={top_rule['confidence']:.4f}, lift={top_rule['lift']:.4f})"
        )

    lines.append("")
    lines.append("## Decision Making / Conclusion")
    lines.append(
        "- Use the best regression model to estimate salary ranges for new job postings and benchmark compensation strategy."
    )
    lines.append(
        "- Use the best classification model to prioritize job postings likely to belong to high-salary segment."
    )
    lines.append(
        "- Use cluster profiles for targeted hiring campaigns (remote-friendly roles, senior-heavy roles, or skill-intensive roles)."
    )
    lines.append(
        "- Use association rules to design learning roadmaps by bundling co-occurring skills."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("- regression_results.csv")
    lines.append("- classification_results.csv")
    lines.append("- clustering_scores.csv")
    lines.append("- cluster_profile.csv")
    lines.append("- association_rules_top20.csv")
    lines.append("- regression_comparison.png")
    lines.append("- classification_comparison.png")
    lines.append("- cluster_scatter_pca.png")
    lines.append("- top_skill_support.png")
    lines.append("- detail_salary_distribution.png")
    lines.append("- detail_salary_by_experience_boxplot.png")
    lines.append("- detail_remote_ratio_vs_salary.png")
    lines.append("- detail_regression_actual_vs_pred.png")
    lines.append("- detail_regression_residual_distribution.png")
    lines.append("- detail_classification_confusion_matrix.png")
    lines.append("- detail_classification_roc_curves.png")
    lines.append("- detail_clustering_silhouette_by_k.png")
    lines.append("- detail_cluster_size_distribution.png")
    lines.append("- detail_association_support_confidence_lift.png")

    report_path = output_dir / "PROJECT_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "ai_job_dataset_merged_cleaned.csv"
    output_dir = base_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError("Missing ai_job_dataset_merged_cleaned.csv. Run Process.py first.")

    sns.set_theme(style="whitegrid")

    df = pd.read_csv(input_path)

    features, numeric_cols, categorical_cols = build_feature_sets(df.copy())

    reg_results, best_reg_model, reg_diagnostics = regression_experiment(
        df, features, numeric_cols, categorical_cols
    )
    cls_results, best_cls_model, cls_diagnostics = classification_experiment(
        df, features, numeric_cols, categorical_cols
    )
    clustering_scores, best_k, clustered_df = clustering_experiment(df.copy(), output_dir)
    assoc_rules, _ = association_experiment(df, output_dir)

    reg_results.to_csv(output_dir / "regression_results.csv", index=False)
    cls_results.to_csv(output_dir / "classification_results.csv", index=False)

    save_model_plots(reg_results, cls_results, output_dir)
    save_detailed_visualizations(
        df=df,
        output_dir=output_dir,
        reg_results=reg_results,
        cls_results=cls_results,
        clustering_scores=clustering_scores,
        clustered_df=clustered_df,
        reg_diagnostics=reg_diagnostics,
        cls_diagnostics=cls_diagnostics,
        assoc_rules=assoc_rules,
    )

    write_report(output_dir, reg_results, cls_results, clustering_scores, best_k, assoc_rules)

    print("Topic analysis completed.")
    print(f"Best regression model: {best_reg_model}")
    print(f"Best classification model: {best_cls_model}")
    print(f"Artifacts folder: {output_dir}")


if __name__ == "__main__":
    main()
