"""
Produces the single, deployable artifact: one scikit-learn Pipeline object that
does ALL preprocessing (imputation, scaling, one-hot encoding) and model training
(tuned XGBoost) end-to-end, so a fresh clone can load it and score new
transactions with no manual preprocessing step required.

This is the model recommended in the report: tuned XGBoost, decision threshold
0.08 on isotonic-calibrated probabilities (the threshold itself is documented
here, not baked into the pipeline, since scikit-learn pipelines output
probabilities, not thresholded decisions - keeping the threshold as an explicit,
documented constant is safer than hiding it inside a custom estimator).
"""
import ast
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
import xgboost as xgb

from models import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, TIME_COL

FINAL_THRESHOLD = 0.08  # isotonic-calibrated, cost-optimal (see report Section 3 & 5)


def load_best_params():
    with open('outputs/tables/best_xgb_params.txt') as f:
        return ast.literal_eval(f.read())


def build_final_pipeline(params, pos_weight):
    return Pipeline([
        ('preprocess', build_preprocessor()),
        ('model', xgb.XGBClassifier(**params, scale_pos_weight=pos_weight,
                                     eval_metric='aucpr', random_state=42, n_jobs=-1)),
    ])


if __name__ == '__main__':
    df = pd.read_csv('data/processed/model_data.csv', parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    pos_weight = (y == 0).sum() / (y == 1).sum()

    params = load_best_params()
    pipeline = build_final_pipeline(params, pos_weight)
    pipeline.fit(X, y)

    joblib.dump(pipeline, 'outputs/final_fraud_pipeline.joblib')

    metadata = {
        'model': 'XGBoost (Optuna-tuned)',
        'decision_threshold_on_calibrated_score': FINAL_THRESHOLD,
        'note': 'Pipeline outputs raw probabilities via .predict_proba(X)[:, 1]. '
                'Apply isotonic calibration (see src/calibration.py) before thresholding '
                'at FINAL_THRESHOLD for a production decision.',
        'features_numeric': NUMERIC_FEATURES,
        'features_categorical': CATEGORICAL_FEATURES,
        'best_params': params,
        'training_rows': int(len(df)),
        'training_date_range': [str(df[TIME_COL].min()), str(df[TIME_COL].max())],
    }
    with open('outputs/final_pipeline_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print('Saved outputs/final_fraud_pipeline.joblib and outputs/final_pipeline_metadata.json')

    # reload check - does the saved pipeline work with no changes needed?
    reloaded = joblib.load('outputs/final_fraud_pipeline.joblib')
    sample = X.sample(5, random_state=1)
    print('\nReload check - predict_proba on 5 sample rows:')
    print(reloaded.predict_proba(sample)[:, 1])
