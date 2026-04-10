from pathlib import Path
import pandas as pd
from run_topic_analysis import build_feature_sets, hyperparameter_tuning

base_dir = Path(__file__).resolve().parent
input_path = base_dir / 'ai_job_dataset_merged_cleaned.csv'
output_dir = base_dir / 'results'
output_dir.mkdir(parents=True, exist_ok=True)

print(f'Loading dataset from {input_path}')
df = pd.read_csv(input_path)
features, numeric_cols, categorical_cols = build_feature_sets(df.copy())
print('Running hyperparameter tuning...')
hyperparameter_tuning(df, features, numeric_cols, categorical_cols, output_dir)
print('Done')
