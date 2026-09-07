"""
Trains the required model ladder (regularised logistic regression, random forest,
XGBoost) on the SAME purged/blocked time-series folds, and reports both PR-AUC
(primary metric, since positives are rare - ~8.4%) and ROC-AUC (for contrast,
to show why it overstates performance on imbalanced data).

Each model uses class weighting to handle the imbalance at this baseline-comparison
stage. Deeper imbalance-handling comparisons (class weighting vs SMOTE, and the
leakage effect of resampling outside folds) are done separately in imbalance.py.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from cv import purged_blocked_time_series_split

NUMERIC_FEATURES = [
    'amount_clean', 'balance_after', 'txn_count_7d', 'txn_count_30d',
    'total_amount_30d', 'avg_amount_30d', 'cashout_ratio_30d', 'debit_ratio_30d',
    'hours_since_last_txn', 'is_night_txn', 'day_of_week', 'account_age_days',
    'distinct_devices_30d', 'distinct_agents_30d', 'device_shared_flag',
]
CATEGORICAL_FEATURES = ['segment', 'region']
TARGET = 'is_fraud'
TIME_COL = 'txn_time_clean'


def build_preprocessor():
    return ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
    ])


def get_models(pos_weight: float):
    """pos_weight = (n_negative / n_positive) on the training fold, used for
    scale_pos_weight (XGBoost) and class_weight (sklearn models)."""
    return {
        'logistic_regression': Pipeline([
            ('prep', build_preprocessor()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced',
                                        C=1.0, random_state=42)),
        ]),
        'random_forest': Pipeline([
            ('prep', build_preprocessor()),
            ('clf', RandomForestClassifier(n_estimators=300, max_depth=10,
                                            class_weight='balanced',
                                            random_state=42, n_jobs=-1)),
        ]),
        'xgboost': Pipeline([
            ('prep', build_preprocessor()),
            ('clf', xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                       scale_pos_weight=pos_weight, eval_metric='aucpr',
                                       random_state=42, n_jobs=-1)),
        ]),
    }


def run_comparison(df: pd.DataFrame, n_splits: int = 5, purge_days: int = 30):
    results = []
    fold_id = 0
    for train_idx, test_idx in purged_blocked_time_series_split(df[TIME_COL], n_splits, purge_days):
        fold_id += 1
        X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = X_train[TARGET], X_test[TARGET]
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        models = get_models(pos_weight)
        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pr_auc = average_precision_score(y_test, proba)
            roc_auc = roc_auc_score(y_test, proba)
            results.append({
                'fold': fold_id, 'model': name,
                'pr_auc': pr_auc, 'roc_auc': roc_auc,
                'n_train': len(X_train), 'n_test': len(X_test),
            })
            print(f"fold {fold_id} | {name:20s} | PR-AUC={pr_auc:.4f} | ROC-AUC={roc_auc:.4f}")
    return pd.DataFrame(results)


if __name__ == '__main__':
    df = pd.read_csv('data/processed/model_data.csv', parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    results = run_comparison(df)
    results.to_csv('outputs/tables/model_comparison_by_fold.csv', index=False)

    summary = results.groupby('model')[['pr_auc', 'roc_auc']].agg(['mean', 'std']).round(4)
    print('\n=== Summary across folds ===')
    print(summary)
    summary.to_csv('outputs/tables/model_comparison_summary.csv')
