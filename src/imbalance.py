"""
Compares two ways of handling the ~8.4% positive class rate:
  (A) class weighting (scale_pos_weight / class_weight='balanced') - already
      used as the baseline throughout models.py and cost.py
  (B) SMOTE oversampling, applied CORRECTLY - fit only on each fold's training
      data, never on the test fold

Then deliberately demonstrates the leakage that the assignment asks us to prove:
  (C) SMOTE applied to the WHOLE dataset BEFORE splitting into folds - synthetic
      points generated from a test-fold row's neighbours can leak into training,
      inflating the offline score in a way that would not hold in production.

All three are evaluated with the same XGBoost architecture and the same purged
time-series folds, so the only thing that changes is how (and when) resampling
happens.
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

from cv import purged_blocked_time_series_split
from models import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, TIME_COL, build_preprocessor


def fit_transform_features(X_train, X_test):
    prep = build_preprocessor()
    Xtr = prep.fit_transform(X_train)
    Xte = prep.transform(X_test)
    # SMOTE needs dense arrays
    if hasattr(Xtr, 'toarray'):
        Xtr = Xtr.toarray()
        Xte = Xte.toarray()
    return Xtr, Xte


def run_class_weighting(df, n_splits=5, purge_days=30):
    scores = []
    for train_idx, test_idx in purged_blocked_time_series_split(df[TIME_COL], n_splits, purge_days):
        X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = X_train[TARGET], X_test[TARGET]
        Xtr, Xte = fit_transform_features(X_train, X_test)
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   scale_pos_weight=pos_weight, eval_metric='aucpr',
                                   random_state=42, n_jobs=-1)
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        scores.append({'method': 'class_weighting',
                        'pr_auc': average_precision_score(y_test, proba),
                        'roc_auc': roc_auc_score(y_test, proba)})
    return scores


def run_smote_correct(df, n_splits=5, purge_days=30):
    """SMOTE fit ONLY on each fold's training data - the correct way."""
    scores = []
    for train_idx, test_idx in purged_blocked_time_series_split(df[TIME_COL], n_splits, purge_days):
        X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = X_train[TARGET], X_test[TARGET]
        Xtr, Xte = fit_transform_features(X_train, X_test)

        sm = SMOTE(random_state=42)
        Xtr_res, ytr_res = sm.fit_resample(Xtr, y_train)

        model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   eval_metric='aucpr', random_state=42, n_jobs=-1)
        model.fit(Xtr_res, ytr_res)
        proba = model.predict_proba(Xte)[:, 1]
        scores.append({'method': 'smote_correct_inside_fold',
                        'pr_auc': average_precision_score(y_test, proba),
                        'roc_auc': roc_auc_score(y_test, proba)})
    return scores


def run_smote_leaky(df, n_splits=5, purge_days=30):
    """DELIBERATE LEAKAGE DEMO: SMOTE fit on the whole dataset BEFORE splitting.
    Synthetic minority points are interpolated between real minority points and
    their nearest neighbours - if that happens before the time split, a synthetic
    row placed in the 'training' fold can be built from a real neighbour that
    belongs to the 'test' fold, letting test-fold information leak into training."""
    prep = build_preprocessor()
    X_all = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_all = df[TARGET]
    Xall_t = prep.fit_transform(X_all)
    if hasattr(Xall_t, 'toarray'):
        Xall_t = Xall_t.toarray()

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(Xall_t, y_all)
    # keep the same row count relationship: the resampled data no longer has a
    # clean row-to-time mapping, so for this leakage DEMONSTRATION we mimic the
    # common real-world mistake of resampling first, then doing a plain
    # (non-purged) time-agnostic split over the resampled data.
    n = len(X_res)
    split = int(n * 0.8)
    Xtr, Xte = X_res[:split], X_res[split:]
    ytr, yte = y_res[:split], y_res[split:]

    model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                               eval_metric='aucpr', random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    return [{'method': 'smote_leaky_before_split',
              'pr_auc': average_precision_score(yte, proba),
              'roc_auc': roc_auc_score(yte, proba)}]


if __name__ == '__main__':
    df = pd.read_csv('data/processed/model_data.csv', parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    all_scores = []
    print('Running class weighting (correct, per-fold)...')
    all_scores += run_class_weighting(df)
    print('Running SMOTE correct (inside each fold)...')
    all_scores += run_smote_correct(df)
    print('Running SMOTE leaky (whole-dataset resample before split)...')
    all_scores += run_smote_leaky(df)

    result = pd.DataFrame(all_scores)
    print('\n=== Per-fold / per-run scores ===')
    print(result)

    summary = result.groupby('method')[['pr_auc', 'roc_auc']].mean().round(4)
    print('\n=== Mean scores by method ===')
    print(summary)
    result.to_csv('outputs/tables/imbalance_comparison_raw.csv', index=False)
    summary.to_csv('outputs/tables/imbalance_comparison_summary.csv')
