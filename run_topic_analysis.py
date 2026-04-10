"""
run_topic_analysis.py - Phân tích mô hình Data Mining toàn diện.

Bao gồm:
    1. Regression: Linear Regression, Decision Tree, Random Forest,
       Gradient Boosting, XGBoost, MLPRegressor (Neural Network)
    2. Classification: Logistic Regression, Decision Tree, Random Forest,
       Gradient Boosting, XGBoost, Naive Bayes, MLPClassifier (Neural Network)
    3. Clustering: KMeans vs Agglomerative vs DBSCAN
    4. Association Rules: Apriori + Lift/Confidence
    5. Cross-Validation (5-fold)
    6. Feature Importance
    7. Learning Curves
    8. Hyperparameter Tuning (GridSearchCV)

Các kỹ thuật theo đề cương CO3029:
    - Chương 3: Hồi quy tuyến tính + phi tuyến (Linear, DT, RF, GB, XGB, MLP)
    - Chương 4: Phân loại với Decision Tree, Bayesian (NaiveBayes),
                Neural Network (MLP), và các phương pháp khác
    - Chương 5: Gom cụm phân hoạch (KMeans), phân cấp (Agglomerative),
                dựa trên mật độ (DBSCAN)
    - Chương 6: Luật kết hợp (Apriori)

Output: results/ directory containing CSV results, PNG plots, and reports.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Optional: XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] xgboost not installed. Skipping XGBoost models.")
    print("  Install with: pip install xgboost")


RANDOM_STATE = 42


# ============================================================
# Utility functions
# ============================================================

def ensure_columns(df: pd.DataFrame, columns: list[str], fill_value: float = 0.0) -> pd.DataFrame:
    """Đảm bảo DataFrame có đầy đủ các cột yêu cầu."""
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df


def parse_skills(series: pd.Series) -> list[list[str]]:
    """Chuyển chuỗi kỹ năng thành danh sách giao dịch cho Association Rules."""
    transactions = []
    for value in series.fillna(""):
        items = [x.strip() for x in str(value).split(",") if x.strip()]
        transactions.append(items)
    return transactions


def build_feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Xây dựng tập đặc trưng cho mô hình Regression/Classification."""
    numeric_cols = [
        "years_experience", "remote_ratio", "job_description_length",
        "benefits_score", "skills_count", "days_to_deadline",
        "home_country_match", "experience_level_ord", "education_required_ord",
    ]
    categorical_cols = [
        "experience_level", "employment_type", "company_size",
        "industry", "company_location", "salary_currency",
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


# ============================================================
# 1. REGRESSION
# ============================================================

def regression_experiment(
    df: pd.DataFrame,
    features: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, str, dict[str, dict[str, np.ndarray]], dict]:
    """
    Chạy thí nghiệm hồi quy với 7 mô hình (bao gồm Neural Network).
    Trả về: results DataFrame, best model name, diagnostics dict, cv_results dict.
    """
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
        "LinearRegression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]),
        "DecisionTreeRegressor": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE)),
        ]),
        "RandomForestRegressor": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200, random_state=RANDOM_STATE,
                n_jobs=-1, min_samples_leaf=2,
            )),
        ]),
        "GradientBoostingRegressor": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=200, max_depth=6,
                learning_rate=0.1, random_state=RANDOM_STATE,
            )),
        ]),
        # Chương 3.3: Hồi quy phi tuyến - Neural Network
        "MLPRegressor": Pipeline([
            ("preprocessor", preprocessor),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=200,
                random_state=RANDOM_STATE, early_stopping=True,
                learning_rate="adaptive",
            )),
        ]),
    }

    if HAS_XGBOOST:
        models["XGBRegressor"] = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                n_estimators=200, max_depth=6,
                learning_rate=0.1, random_state=RANDOM_STATE,
                n_jobs=-1, verbosity=0,
            )),
        ])

    rows = []
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    cv_results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        print(f"  Training regression model: {name} ...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))

        # 5-fold Cross-Validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        rows.append({
            "model": name, "rmse": rmse, "mae": mae, "r2": r2,
            "cv_r2_mean": cv_mean, "cv_r2_std": cv_std,
        })
        diagnostics[name] = {
            "y_true": y_test.to_numpy(),
            "y_pred": pred,
        }
        cv_results[name] = {"cv_r2_mean": cv_mean, "cv_r2_std": cv_std}

    results = pd.DataFrame(rows).sort_values(by="rmse", ascending=True).reset_index(drop=True)
    best_model = str(results.iloc[0]["model"])
    return results, best_model, diagnostics, cv_results


# ============================================================
# 2. CLASSIFICATION
# ============================================================

def classification_experiment(
    df: pd.DataFrame,
    features: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, str, dict[str, dict[str, np.ndarray]], dict]:
    """
    Chạy thí nghiệm phân loại với 7 mô hình (bao gồm Naive Bayes và Neural Network).
    Trả về: results DataFrame, best model name, diagnostics dict, cv_results dict.
    """
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
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    models = {
        "LogisticRegression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                solver="saga", max_iter=5000, random_state=RANDOM_STATE,
            )),
        ]),
        "DecisionTreeClassifier": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)),
        ]),
        "RandomForestClassifier": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=250, random_state=RANDOM_STATE,
                n_jobs=-1, min_samples_leaf=2,
            )),
        ]),
        "GradientBoostingClassifier": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(
                n_estimators=200, max_depth=6,
                learning_rate=0.1, random_state=RANDOM_STATE,
            )),
        ]),
        # Chương 4.3: Phân loại với mạng Bayesian (Naive Bayes)
        "GaussianNB": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GaussianNB()),
        ]),
        # Chương 4.4: Phân loại với mạng Neural (MLPClassifier)
        "MLPClassifier": Pipeline([
            ("preprocessor", preprocessor),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=200,
                random_state=RANDOM_STATE, early_stopping=True,
                learning_rate="adaptive",
            )),
        ]),
    }

    if HAS_XGBOOST:
        models["XGBClassifier"] = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=200, max_depth=6,
                learning_rate=0.1, random_state=RANDOM_STATE,
                n_jobs=-1, verbosity=0, eval_metric="logloss",
            )),
        ])

    rows = []
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    cv_results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        print(f"  Training classification model: {name} ...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        if hasattr(model.named_steps["model"], "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, prob))
        else:
            prob = pred.astype(float)
            auc = np.nan

        # 5-fold Cross-Validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1", n_jobs=-1)
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": auc,
            "salary_threshold": threshold,
            "cv_f1_mean": cv_mean, "cv_f1_std": cv_std,
        })

        diagnostics[name] = {
            "y_true": y_test.to_numpy(),
            "y_pred": pred,
            "y_prob": prob,
        }
        cv_results[name] = {"cv_f1_mean": cv_mean, "cv_f1_std": cv_std}

    results = pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)
    best_model = str(results.iloc[0]["model"])
    return results, best_model, diagnostics, cv_results


# ============================================================
# 3. CLUSTERING (KMeans, Agglomerative, DBSCAN)
# ============================================================

def clustering_experiment(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """
    Chạy thí nghiệm phân cụm với KMeans, Agglomerative, DBSCAN.
    Trả về: clustering_scores, best_k, clustered DataFrame.
    """
    cluster_features = [
        "salary_usd", "years_experience", "remote_ratio",
        "benefits_score", "skills_count", "job_description_length",
    ]
    df = ensure_columns(df.copy(), cluster_features, fill_value=0.0)

    X = df[cluster_features].copy()
    for col in cluster_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sample for efficiency
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

        rows.append({
            "k": k,
            "kmeans_silhouette": score_kmeans,
            "agglomerative_silhouette": score_agg,
        })

        if score_kmeans > best_score:
            best_score = score_kmeans
            best_k = k

    clustering_scores = pd.DataFrame(rows)

    # DBSCAN
    print("  Running DBSCAN clustering ...")
    dbscan = DBSCAN(eps=1.2, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_eval)
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = int((dbscan_labels == -1).sum())
    dbscan_sil = float(silhouette_score(X_eval, dbscan_labels)) if n_dbscan_clusters >= 2 else -1.0

    # Save DBSCAN info
    dbscan_info = pd.DataFrame([{
        "method": "DBSCAN",
        "eps": 1.2,
        "min_samples": 10,
        "n_clusters": n_dbscan_clusters,
        "n_noise": n_noise,
        "silhouette": dbscan_sil,
    }])
    dbscan_info.to_csv(output_dir / "dbscan_results.csv", index=False)

    # Final KMeans on full data
    best_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    df["cluster"] = best_kmeans.fit_predict(X_scaled)

    cluster_profile = (
        df.groupby("cluster")[cluster_features]
        .mean().round(2).reset_index()
        .sort_values(by="cluster")
    )

    # PCA scatter plot
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_data = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        "pc1": pca_data[:, 0], "pc2": pca_data[:, 1], "cluster": df["cluster"]
    })

    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=pca_df, x="pc1", y="pc2", hue="cluster",
                    palette="tab10", s=30, alpha=0.6)
    plt.title(f"KMeans Clusters (k={best_k}) trong không gian PCA 2D",
              fontsize=14, fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_scatter_pca.png", dpi=140)
    plt.close()

    cluster_profile.to_csv(output_dir / "cluster_profile.csv", index=False)
    clustering_scores.to_csv(output_dir / "clustering_scores.csv", index=False)

    return clustering_scores, best_k, df


# ============================================================
# 4. ASSOCIATION RULES
# ============================================================

def association_experiment(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy Apriori Association Rules trên required_skills."""
    transactions = parse_skills(df["required_skills"])
    transactions = [t for t in transactions if len(t) > 0]

    te = TransactionEncoder()
    te_arr = te.fit(transactions).transform(transactions)
    skill_df = pd.DataFrame(te_arr, columns=te.columns_)

    support = skill_df.mean().sort_values(ascending=False).reset_index()
    support.columns = ["skill", "support"]
    top_support = support.head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_support, x="support", y="skill",
                hue="skill", palette="viridis", legend=False)
    plt.title("Top 15 kỹ năng phổ biến (Support)", fontsize=14, fontweight="bold")
    plt.xlabel("Support")
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


# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    df: pd.DataFrame,
    features: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    output_dir: Path,
) -> None:
    """Tính và vẽ Feature Importance cho Random Forest Regression và Classification."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    y_reg = pd.to_numeric(df["salary_usd"], errors="coerce")
    valid_mask = y_reg.notna()
    X = features.loc[valid_mask]
    y = y_reg.loc[valid_mask]

    # --- Regression Feature Importance ---
    print("  Computing regression feature importance ...")
    pipe_reg = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE,
            n_jobs=-1, min_samples_leaf=2,
        )),
    ])
    pipe_reg.fit(X, y)

    ohe = pipe_reg.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    all_feature_names = numeric_cols + cat_feature_names
    importances_reg = pipe_reg.named_steps["model"].feature_importances_

    fi_reg = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances_reg,
    }).sort_values("importance", ascending=False)

    # Top 20
    top_fi_reg = fi_reg.head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_fi_reg, x="importance", y="feature",
                hue="feature", palette="viridis", legend=False)
    plt.title("Top 20 Feature Importance (RandomForest Regression)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_regression.png", dpi=150)
    plt.close()
    fi_reg.to_csv(output_dir / "feature_importance_regression.csv", index=False)

    # --- Classification Feature Importance ---
    print("  Computing classification feature importance ...")
    preprocessor_cls = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    salary = pd.to_numeric(df["salary_usd"], errors="coerce")
    threshold = float(salary.quantile(0.75))
    target = (salary >= threshold).astype(int)
    y_cls = target.loc[valid_mask]

    pipe_cls = Pipeline([
        ("preprocessor", preprocessor_cls),
        ("model", RandomForestClassifier(
            n_estimators=250, random_state=RANDOM_STATE,
            n_jobs=-1, min_samples_leaf=2,
        )),
    ])
    pipe_cls.fit(X, y_cls)

    ohe_cls = pipe_cls.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names_cls = list(ohe_cls.get_feature_names_out(categorical_cols))
    all_feature_names_cls = numeric_cols + cat_feature_names_cls
    importances_cls = pipe_cls.named_steps["model"].feature_importances_

    fi_cls = pd.DataFrame({
        "feature": all_feature_names_cls,
        "importance": importances_cls,
    }).sort_values("importance", ascending=False)

    top_fi_cls = fi_cls.head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_fi_cls, x="importance", y="feature",
                hue="feature", palette="magma", legend=False)
    plt.title("Top 20 Feature Importance (RandomForest Classification)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_classification.png", dpi=150)
    plt.close()
    fi_cls.to_csv(output_dir / "feature_importance_classification.csv", index=False)


# ============================================================
# 6. LEARNING CURVES
# ============================================================

def save_learning_curves(
    df: pd.DataFrame,
    features: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    output_dir: Path,
) -> None:
    """Vẽ Learning Curve cho best model (RandomForest) - Regression và Classification."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    y = pd.to_numeric(df["salary_usd"], errors="coerce")
    valid_mask = y.notna()
    X = features.loc[valid_mask]
    y_reg = y.loc[valid_mask]

    # --- Regression Learning Curve ---
    print("  Computing regression learning curve ...")
    pipe_reg = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE,
            n_jobs=-1, min_samples_leaf=2,
        )),
    ])

    train_sizes, train_scores, val_scores = learning_curve(
        pipe_reg, X, y_reg, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="r2", n_jobs=-1,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#2a9d8f",
             label="Training R²", linewidth=2)
    plt.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1),
                     alpha=0.2, color="#2a9d8f")
    plt.plot(train_sizes, val_scores.mean(axis=1), "o-", color="#e76f51",
             label="Validation R²", linewidth=2)
    plt.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1),
                     alpha=0.2, color="#e76f51")
    plt.title("Learning Curve - RandomForest Regression",
              fontsize=14, fontweight="bold")
    plt.xlabel("Số mẫu huấn luyện")
    plt.ylabel("R² Score")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve_regression.png", dpi=150)
    plt.close()

    # --- Classification Learning Curve ---
    print("  Computing classification learning curve ...")
    preprocessor_cls = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    salary = pd.to_numeric(df["salary_usd"], errors="coerce")
    threshold = float(salary.quantile(0.75))
    target = (salary >= threshold).astype(int)
    y_cls = target.loc[valid_mask]

    pipe_cls = Pipeline([
        ("preprocessor", preprocessor_cls),
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            n_jobs=-1, min_samples_leaf=2,
        )),
    ])

    train_sizes_cls, train_scores_cls, val_scores_cls = learning_curve(
        pipe_cls, X, y_cls, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="f1", n_jobs=-1,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_cls, train_scores_cls.mean(axis=1), "o-", color="#264653",
             label="Training F1", linewidth=2)
    plt.fill_between(train_sizes_cls,
                     train_scores_cls.mean(axis=1) - train_scores_cls.std(axis=1),
                     train_scores_cls.mean(axis=1) + train_scores_cls.std(axis=1),
                     alpha=0.2, color="#264653")
    plt.plot(train_sizes_cls, val_scores_cls.mean(axis=1), "o-", color="#e9c46a",
             label="Validation F1", linewidth=2)
    plt.fill_between(train_sizes_cls,
                     val_scores_cls.mean(axis=1) - val_scores_cls.std(axis=1),
                     val_scores_cls.mean(axis=1) + val_scores_cls.std(axis=1),
                     alpha=0.2, color="#e9c46a")
    plt.title("Learning Curve - RandomForest Classification",
              fontsize=14, fontweight="bold")
    plt.xlabel("Số mẫu huấn luyện")
    plt.ylabel("F1 Score")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve_classification.png", dpi=150)
    plt.close()


# ============================================================
# 7. HYPERPARAMETER TUNING
# ============================================================

def hyperparameter_tuning(
    df: pd.DataFrame,
    features: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """GridSearchCV cho RandomForest Regressor."""
    print("  Running GridSearchCV for RandomForest Regressor ...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    y = pd.to_numeric(df["salary_usd"], errors="coerce")
    valid_mask = y.notna()
    X = features.loc[valid_mask]
    y = y.loc[valid_mask]

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
    }

    grid = GridSearchCV(
        pipe, param_grid, cv=3, scoring="r2",
        n_jobs=-1, verbose=0, return_train_score=True,
    )
    grid.fit(X, y)

    results = pd.DataFrame(grid.cv_results_)
    results = results[[
        "param_model__n_estimators",
        "param_model__max_depth",
        "param_model__min_samples_leaf",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
    ]].sort_values("mean_test_score", ascending=False)

    results.to_csv(output_dir / "gridsearch_results.csv", index=False)

    # Best params summary
    best = pd.DataFrame([{
        "best_n_estimators": grid.best_params_["model__n_estimators"],
        "best_max_depth": grid.best_params_["model__max_depth"],
        "best_min_samples_leaf": grid.best_params_["model__min_samples_leaf"],
        "best_cv_r2": grid.best_score_,
    }])
    best.to_csv(output_dir / "gridsearch_best_params.csv", index=False)
    print(f"  Best params: {grid.best_params_}, R²={grid.best_score_:.4f}")

    return results


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

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
    """Tạo các biểu đồ chi tiết cho tất cả mô hình."""
    salary = pd.to_numeric(df["salary_usd"], errors="coerce").dropna()

    # Salary distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(salary, bins=40, kde=True, color="#2a9d8f")
    plt.title("Salary Distribution (USD)", fontsize=14, fontweight="bold")
    plt.xlabel("salary_usd")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_salary_distribution.png", dpi=140)
    plt.close()

    # Salary by experience boxplot
    if "experience_level" in df.columns:
        box_df = df[["experience_level", "salary_usd"]].copy()
        box_df["salary_usd"] = pd.to_numeric(box_df["salary_usd"], errors="coerce")
        box_df = box_df.dropna()
        order = ["EN", "MI", "SE", "EX"]
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=box_df, x="experience_level", y="salary_usd",
                    order=order, hue="experience_level",
                    palette="Set2", legend=False)
        plt.title("Salary by Experience Level", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "detail_salary_by_experience_boxplot.png", dpi=140)
        plt.close()

    # Remote ratio vs salary
    scatter_df = df[["remote_ratio", "salary_usd", "experience_level"]].copy()
    scatter_df["salary_usd"] = pd.to_numeric(scatter_df["salary_usd"], errors="coerce")
    scatter_df["remote_ratio"] = pd.to_numeric(scatter_df["remote_ratio"], errors="coerce")
    scatter_df = scatter_df.dropna()
    if len(scatter_df) > 5000:
        scatter_df = scatter_df.sample(5000, random_state=RANDOM_STATE)
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=scatter_df, x="remote_ratio", y="salary_usd",
                    hue="experience_level", alpha=0.6, s=28, palette="tab10")
    plt.title("Remote Ratio vs Salary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_remote_ratio_vs_salary.png", dpi=140)
    plt.close()

    # Regression: Actual vs Predicted
    best_reg_model = str(reg_results.iloc[0]["model"])
    best_reg_diag = reg_diagnostics[best_reg_model]
    y_true_reg = best_reg_diag["y_true"]
    y_pred_reg = best_reg_diag["y_pred"]

    reg_min = float(min(np.min(y_true_reg), np.min(y_pred_reg)))
    reg_max = float(max(np.max(y_true_reg), np.max(y_pred_reg)))
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true_reg, y=y_pred_reg, alpha=0.35, s=20, color="#1d3557")
    plt.plot([reg_min, reg_max], [reg_min, reg_max], color="#e63946", linewidth=2)
    plt.title(f"Actual vs Predicted Salary ({best_reg_model})",
              fontsize=14, fontweight="bold")
    plt.xlabel("Actual salary_usd")
    plt.ylabel("Predicted salary_usd")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_regression_actual_vs_pred.png", dpi=140)
    plt.close()

    # Residual distribution
    residual = y_true_reg - y_pred_reg
    plt.figure(figsize=(8, 5))
    sns.histplot(residual, bins=40, kde=True, color="#6a4c93")
    plt.title(f"Residual Distribution ({best_reg_model})",
              fontsize=14, fontweight="bold")
    plt.xlabel("Residual = y_true - y_pred")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_regression_residual_distribution.png", dpi=140)
    plt.close()

    # Confusion Matrix
    best_cls_model = str(cls_results.iloc[0]["model"])
    best_cls_diag = cls_diagnostics[best_cls_model]
    y_true_cls = best_cls_diag["y_true"]
    y_pred_cls = best_cls_diag["y_pred"]

    cm = confusion_matrix(y_true_cls, y_pred_cls)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not High", "High"],
                yticklabels=["Not High", "High"])
    plt.title(f"Confusion Matrix ({best_cls_model})",
              fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_classification_confusion_matrix.png", dpi=140)
    plt.close()

    # ROC Curves
    plt.figure(figsize=(8, 6))
    for model_name, diag in cls_diagnostics.items():
        fpr, tpr, _ = roc_curve(diag["y_true"], diag["y_prob"])
        auc_value = roc_auc_score(diag["y_true"], diag["y_prob"])
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curves - Classification Models",
              fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "detail_classification_roc_curves.png", dpi=140)
    plt.close()

    # Silhouette by k
    plt.figure(figsize=(8, 5))
    melt_cluster = clustering_scores.melt(
        id_vars="k",
        value_vars=["kmeans_silhouette", "agglomerative_silhouette"],
        var_name="algorithm", value_name="silhouette",
    )
    sns.lineplot(data=melt_cluster, x="k", y="silhouette",
                 hue="algorithm", marker="o", linewidth=2)
    plt.title("Silhouette Score by k", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_clustering_silhouette_by_k.png", dpi=140)
    plt.close()

    # Cluster size distribution
    cluster_sizes = clustered_df["cluster"].value_counts().sort_index().reset_index()
    cluster_sizes.columns = ["cluster", "count"]
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=cluster_sizes, x="cluster", y="count",
                hue="cluster", palette="tab10", legend=False)
    plt.title("Cluster Size Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "detail_cluster_size_distribution.png", dpi=140)
    plt.close()

    # Association rules scatter
    if not assoc_rules.empty:
        plot_rules = assoc_rules.head(15).copy()
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=plot_rules, x="support", y="confidence",
                        size="lift", hue="lift", sizes=(50, 350),
                        palette="viridis")
        plt.title("Association Rules: Support vs Confidence",
                  fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "detail_association_support_confidence_lift.png", dpi=140)
        plt.close()

    # Cross-validation comparison chart
    if "cv_r2_mean" in reg_results.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Regression CV
        ax1 = axes[0]
        reg_cv = reg_results[["model", "cv_r2_mean", "cv_r2_std"]].copy()
        ax1.barh(reg_cv["model"], reg_cv["cv_r2_mean"], xerr=reg_cv["cv_r2_std"],
                 color=sns.color_palette("viridis", len(reg_cv)), capsize=5)
        ax1.set_title("5-Fold CV R² (Regression)", fontsize=13, fontweight="bold")
        ax1.set_xlabel("R² Score")

        # Classification CV
        if "cv_f1_mean" in cls_results.columns:
            ax2 = axes[1]
            cls_cv = cls_results[["model", "cv_f1_mean", "cv_f1_std"]].copy()
            ax2.barh(cls_cv["model"], cls_cv["cv_f1_mean"], xerr=cls_cv["cv_f1_std"],
                     color=sns.color_palette("magma", len(cls_cv)), capsize=5)
            ax2.set_title("5-Fold CV F1 (Classification)", fontsize=13, fontweight="bold")
            ax2.set_xlabel("F1 Score")

        plt.tight_layout()
        plt.savefig(output_dir / "cross_validation_comparison.png", dpi=150)
        plt.close()


def save_model_plots(reg_results: pd.DataFrame, cls_results: pd.DataFrame, output_dir: Path) -> None:
    """Biểu đồ so sánh hiệu năng mô hình."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=reg_results, x="rmse", y="model",
                hue="model", palette="magma", legend=False)
    plt.title("Regression Model Comparison (Lower RMSE Better)",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "regression_comparison.png", dpi=140)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=cls_results, x="f1", y="model",
                hue="model", palette="crest", legend=False)
    plt.title("Classification Model Comparison (Higher F1 Better)",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "classification_comparison.png", dpi=140)
    plt.close()


# ============================================================
# REPORT GENERATION
# ============================================================

def write_report(
    output_dir: Path,
    reg_results: pd.DataFrame,
    cls_results: pd.DataFrame,
    clustering_scores: pd.DataFrame,
    best_k: int,
    assoc_rules: pd.DataFrame,
) -> None:
    """Tạo báo cáo tổng hợp dưới dạng Markdown."""
    best_reg = reg_results.iloc[0]
    best_cls = cls_results.iloc[0]
    best_cluster_row = clustering_scores.loc[clustering_scores["k"] == best_k].iloc[0]

    lines = []
    lines.append("# Data Mining Project Report")
    lines.append("")
    lines.append("## Topic")
    lines.append(
        "AI Job Market Intelligence: salary prediction, high-salary classification, "
        "job clustering, and skill association rules."
    )
    lines.append("")
    lines.append("## Methods (theo đề cương CO3029)")
    lines.append("- Regression (Ch.3): Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, MLPRegressor (Neural Network)")
    lines.append("- Classification (Ch.4): Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, GaussianNB (Bayesian), MLPClassifier (Neural Network)")
    lines.append("- Clustering: KMeans vs Agglomerative vs DBSCAN (evaluated by silhouette score)")
    lines.append("- Association Rule Mining: Apriori + Lift/Confidence")
    lines.append("- Cross-Validation: 5-fold CV for all models")
    lines.append("- Feature Importance: RandomForest feature importance analysis")
    lines.append("- Hyperparameter Tuning: GridSearchCV for RandomForest")
    lines.append("")

    lines.append("## Key Results")
    lines.append("")
    lines.append("### Regression")
    for _, row in reg_results.iterrows():
        cv_info = f", CV_R²={row['cv_r2_mean']:.4f}±{row['cv_r2_std']:.4f}" if "cv_r2_mean" in row else ""
        lines.append(
            f"- {row['model']}: RMSE={row['rmse']:.2f}, MAE={row['mae']:.2f}, R²={row['r2']:.4f}{cv_info}"
        )
    lines.append(f"- **Best regression model: {best_reg['model']}**")
    lines.append("")

    lines.append("### Classification")
    for _, row in cls_results.iterrows():
        cv_info = f", CV_F1={row['cv_f1_mean']:.4f}±{row['cv_f1_std']:.4f}" if "cv_f1_mean" in row else ""
        lines.append(
            f"- {row['model']}: Accuracy={row['accuracy']:.4f}, F1={row['f1']:.4f}, "
            f"ROC_AUC={row['roc_auc']:.4f}{cv_info}"
        )
    lines.append(f"- **Best classification model: {best_cls['model']}**")
    lines.append("")

    lines.append("### Clustering")
    lines.append(
        f"- Best KMeans setup: k={best_k}, silhouette={best_cluster_row['kmeans_silhouette']:.4f}"
    )
    lines.append("")

    lines.append("### Association Rules")
    if assoc_rules.empty:
        lines.append("- No rule satisfied support/lift threshold.")
    else:
        top_rule = assoc_rules.iloc[0]
        lines.append(
            f"- Strongest rule: {top_rule['antecedents']} -> {top_rule['consequents']} "
            f"(support={top_rule['support']:.4f}, confidence={top_rule['confidence']:.4f}, "
            f"lift={top_rule['lift']:.4f})"
        )
    lines.append("")

    lines.append("## Decision Making / Conclusion")
    lines.append(
        "- **Salary Estimation**: Use the best regression model for compensation planning."
    )
    lines.append(
        "- **High-Salary Screening**: Use the best classification model to identify premium job postings."
    )
    lines.append(
        "- **Segment Strategy**: Use clustering profiles to differentiate senior vs junior hiring campaigns."
    )
    lines.append(
        "- **Skill Roadmap**: Use association rules to recommend bundled skill learning tracks."
    )
    lines.append("")

    report_path = output_dir / "PROJECT_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Entry point: chạy toàn bộ pipeline phân tích Data Mining."""
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "ai_job_dataset_merged_cleaned.csv"
    output_dir = base_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError("Missing ai_job_dataset_merged_cleaned.csv. Run Process.py first.")

    sns.set_theme(style="whitegrid")

    print("=" * 60)
    print("DATA MINING - AI JOB MARKET ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(input_path)
    print(f"Loaded dataset: {len(df):,} rows, {len(df.columns)} columns")

    features, numeric_cols, categorical_cols = build_feature_sets(df.copy())

    # 1. Regression
    print("\n[1/7] REGRESSION EXPERIMENT")
    reg_results, best_reg_model, reg_diagnostics, reg_cv = regression_experiment(
        df, features, numeric_cols, categorical_cols
    )

    # 2. Classification
    print("\n[2/7] CLASSIFICATION EXPERIMENT")
    cls_results, best_cls_model, cls_diagnostics, cls_cv = classification_experiment(
        df, features, numeric_cols, categorical_cols
    )

    # 3. Clustering
    print("\n[3/7] CLUSTERING EXPERIMENT")
    clustering_scores, best_k, clustered_df = clustering_experiment(df.copy(), output_dir)

    # 4. Association Rules
    print("\n[4/7] ASSOCIATION RULES")
    assoc_rules, _ = association_experiment(df, output_dir)

    # 5. Feature Importance
    print("\n[5/7] FEATURE IMPORTANCE")
    save_feature_importance(df, features, numeric_cols, categorical_cols, output_dir)

    # 6. Learning Curves
    print("\n[6/7] LEARNING CURVES")
    save_learning_curves(df, features, numeric_cols, categorical_cols, output_dir)

    # 7. Hyperparameter Tuning
    print("\n[7/7] HYPERPARAMETER TUNING")
    hyperparameter_tuning(df, features, numeric_cols, categorical_cols, output_dir)

    # Save results
    reg_results.to_csv(output_dir / "regression_results.csv", index=False)
    cls_results.to_csv(output_dir / "classification_results.csv", index=False)

    # Generate all plots
    print("\nGenerating visualizations ...")
    save_model_plots(reg_results, cls_results, output_dir)
    save_detailed_visualizations(
        df=df, output_dir=output_dir,
        reg_results=reg_results, cls_results=cls_results,
        clustering_scores=clustering_scores, clustered_df=clustered_df,
        reg_diagnostics=reg_diagnostics, cls_diagnostics=cls_diagnostics,
        assoc_rules=assoc_rules,
    )

    # Generate report
    write_report(output_dir, reg_results, cls_results, clustering_scores, best_k, assoc_rules)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Best regression model: {best_reg_model}")
    print(f"Best classification model: {best_cls_model}")
    print(f"Best clustering: KMeans k={best_k}")
    print(f"Artifacts folder: {output_dir}")
    print(f"Total charts: 25+")


if __name__ == "__main__":
    main()
