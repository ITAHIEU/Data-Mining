# DATA MINING MODEL DETAILS

## 1. Topic and Objective

Topic: AI Job Market Intelligence

Main goals:

- Predict salary_usd for AI job postings (Regression).
- Classify whether a job belongs to high-salary segment (Classification).
- Segment jobs into groups with similar characteristics (Clustering).
- Discover co-occurring skill patterns (Association Rules).

Decision-making orientation:

- Compensation benchmarking.
- High-value job prioritization.
- Segment-based hiring strategy.
- Skill roadmap design.

## 2. Data Used

Input files:

- ai_job_dataset.csv
- ai_job_dataset1.csv

After preprocessing and merge:

- Rows: 30000
- Columns: 26
- Missing values: 0
- Duplicate job_id across merged sources: 15000 (expected because both source files can contain the same job_id, but duplicate pair job_id + source_file is 0)

Reference:

- preprocessing_report.json
- ai_job_dataset_merged_cleaned.csv

## 3. Preprocessing and Feature Engineering

Implemented in Process.py and run_topic_analysis.py.

Main steps:

- Type casting:
  - Numeric: salary_usd, salary_local, years_experience, remote_ratio, job_description_length, benefits_score.
  - Datetime: posting_date, application_deadline.
- Missing value handling:
  - Numeric columns: median imputation.
  - Categorical columns: mode imputation.
- Outlier handling:
  - IQR clipping for key numeric columns.
- Derived features:
  - days_to_deadline = application_deadline - posting_date.
  - skills_count = number of items in required_skills.
  - home_country_match = 1 if company_location equals employee_residence, else 0.
  - experience_level_ord: EN=1, MI=2, SE=3, EX=4.
  - education_required_ord: High School=1, Associate=2, Bachelor=3, Master=4, PhD=5.
- Schema harmonization:
  - Create salary_local in file 1 from salary_usd to align both datasets.

## 4. Tools and Libraries

Language and runtime:

- Python 3.12 in virtual environment .venv.

Core libraries:

- pandas, numpy: data processing.
- scikit-learn: ML models and evaluation.
- mlxtend: Apriori and association_rules.
- matplotlib, seaborn: visualization.

## 5. Feature Set for Modeling

Numeric features:

- years_experience
- remote_ratio
- job_description_length
- benefits_score
- skills_count
- days_to_deadline
- home_country_match
- experience_level_ord
- education_required_ord

Categorical features:

- experience_level
- employment_type
- company_size
- industry
- company_location
- salary_currency

Encoding and scaling:

- Categorical: OneHotEncoder(handle_unknown=ignore).
- Numeric:
  - Regression: passthrough.
  - Classification: StandardScaler.

## 6. Algorithms and Hyperparameters

### 6.1 Regression

Target: salary_usd

Train/test split:

- test_size=0.2
- random_state=42

Models:

- LinearRegression: default params.
- DecisionTreeRegressor: max_depth=12, random_state=42.
- RandomForestRegressor: n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42.

Metrics:

- RMSE (lower is better)
- MAE (lower is better)
- R2 (higher is better)

### 6.2 Classification

Target construction:

- high_salary = 1 if salary_usd >= Q3(salary_usd), else 0.
- salary threshold obtained: 150921.75

Train/test split:

- test_size=0.2
- random_state=42
- stratify=y

Models:

- LogisticRegression: solver=saga, max_iter=5000, random_state=42.
- DecisionTreeClassifier: max_depth=10, random_state=42.
- RandomForestClassifier: n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=42.

Metrics:

- Accuracy
- F1
- ROC_AUC

### 6.3 Clustering

Features used:

- salary_usd, years_experience, remote_ratio, benefits_score, skills_count, job_description_length

Methods:

- KMeans and AgglomerativeClustering, compared for k=2..8.
- Data standardized by StandardScaler.
- Silhouette score used for model selection.
- Best k selected by KMeans silhouette.

Runtime optimization:

- Evaluation sample size: up to 6000 rows for silhouette comparison.
- Final best KMeans fitted on full dataset.

### 6.4 Association Rules

Transactions:

- required_skills split by comma.

Method:

- Apriori(min_support=0.03)
- association_rules(metric=lift, min_threshold=1.1)

Rule quality metrics:

- support
- confidence
- lift

## 7. Quantitative Results

### 7.1 Regression Results

| Model                 |     RMSE |      MAE |     R2 |
| --------------------- | -------: | -------: | -----: |
| RandomForestRegressor | 21820.48 | 16407.77 | 0.8681 |
| LinearRegression      | 23374.96 | 17906.89 | 0.8486 |
| DecisionTreeRegressor | 23789.46 | 17430.66 | 0.8432 |

Conclusion:

- Best regression model is RandomForestRegressor (best RMSE and best R2).

### 7.2 Classification Results

| Model                  | Accuracy |     F1 | ROC_AUC | Threshold |
| ---------------------- | -------: | -----: | ------: | --------: |
| RandomForestClassifier |   0.9267 | 0.8527 |  0.9785 | 150921.75 |
| LogisticRegression     |   0.9265 | 0.8520 |  0.9795 | 150921.75 |
| DecisionTreeClassifier |   0.9195 | 0.8398 |  0.9589 | 150921.75 |

Conclusion:

- Best classification by F1 and accuracy: RandomForestClassifier.
- LogisticRegression has slightly higher ROC_AUC.

### 7.3 Clustering Results

|   k | KMeans silhouette | Agglomerative silhouette |
| --: | ----------------: | -----------------------: |
|   2 |            0.2250 |                   0.2147 |
|   3 |            0.1625 |                   0.1237 |
|   4 |            0.1563 |                   0.1126 |
|   5 |            0.1615 |                   0.0952 |
|   6 |            0.1572 |                   0.0886 |
|   7 |            0.1548 |                   0.0933 |
|   8 |            0.1541 |                   0.0900 |

Best setup:

- KMeans with k=2.

Cluster profile means:

- Cluster 0: salary_usd=86411.24, years_experience=3.21, remote_ratio=49.62.
- Cluster 1: salary_usd=186793.56, years_experience=13.19, remote_ratio=50.33.

Interpretation:

- Cluster 1 is a senior and high-compensation segment.
- Cluster 0 is a junior/mid compensation segment.

### 7.4 Association Rule Results

Top rules (sample):

- TensorFlow -> Python: support=0.0869, confidence=0.4261, lift=1.4285
- Python -> TensorFlow: support=0.0869, confidence=0.2913, lift=1.4285
- PyTorch -> SQL: support=0.0498, confidence=0.2662, lift=1.1545

Interpretation:

- Python and TensorFlow strongly co-occur.
- PyTorch and SQL also show positive association.

## 8. Visual Outputs

Main comparison charts:

- results/regression_comparison.png
- results/classification_comparison.png
- results/cluster_scatter_pca.png
- results/top_skill_support.png

Detailed charts:

- results/detail_salary_distribution.png
- results/detail_salary_by_experience_boxplot.png
- results/detail_remote_ratio_vs_salary.png
- results/detail_regression_actual_vs_pred.png
- results/detail_regression_residual_distribution.png
- results/detail_classification_confusion_matrix.png
- results/detail_classification_roc_curves.png
- results/detail_clustering_silhouette_by_k.png
- results/detail_cluster_size_distribution.png
- results/detail_association_support_confidence_lift.png

## 9. Decision Making and Conclusion

- Salary estimation: use RandomForestRegressor for compensation planning and salary benchmarking.
- High-salary screening: use RandomForestClassifier to prioritize postings likely in high-value segment.
- Strategy by segment: use clustering profiles to separate senior vs non-senior hiring campaigns.
- Skill roadmap: use association rules to recommend bundled skill learning tracks.

## 10. Reproducibility

Run order:

1. python Process.py
2. python run_topic_analysis.py

Produced files:

- Cleaned datasets: ai_job_dataset_cleaned.csv, ai_job_dataset1_cleaned.csv, ai_job_dataset_merged_cleaned.csv
- Numeric/model outputs: results/regression_results.csv, results/classification_results.csv, results/clustering_scores.csv, results/cluster_profile.csv, results/association_rules_top20.csv
- Report: results/PROJECT_REPORT.md
- Detailed model note: MODEL_DETAILS.md
