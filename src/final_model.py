"""
Evaluates the TUNED XGBoost (best Optuna params) on the full 5-fold purged CV -
comparable to the other three models in models.py - then runs the same
cost-threshold and calibration analysis used for the baseline models, so the
tuned model can be judged on equal footing for final selection.
"""
import ast
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

from cv import purged_blocked_time_series_split
from models import build_preprocessor, TARGET, TIME_COL
from cost import total_cost, find_best_threshold
from calibration import isotonic_calibrate, reliability_table


def load_best_params():
    with open('outputs/tables/best_xgb_params.txt') as f:
        return ast.literal_eval(f.read())


def evaluate_tuned_xgb(df, params, n_splits=5, purge_days=30):
    fold_scores = []
    oof_proba = pd.Series(index=df.index, dtype=float)
    covered = pd.Series(False, index=df.index)

    for fold_id, (train_idx, test_idx) in enumerate(
            purged_blocked_time_series_split(df[TIME_COL], n_splits, purge_days), 1):
        X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = X_train[TARGET], X_test[TARGET]
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        prep = build_preprocessor()
        Xtr = prep.fit_transform(X_train)
        Xte = prep.transform(X_test)

        model = xgb.XGBClassifier(**params, scale_pos_weight=pos_weight,
                                   eval_metric='aucpr', random_state=42, n_jobs=-1)
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]

        pr_auc = average_precision_score(y_test, proba)
        roc_auc = roc_auc_score(y_test, proba)
        fold_scores.append({'fold': fold_id, 'model': 'xgboost_tuned', 'pr_auc': pr_auc, 'roc_auc': roc_auc})
        print(f"fold {fold_id} | xgboost_tuned | PR-AUC={pr_auc:.4f} | ROC-AUC={roc_auc:.4f}")

        oof_proba.iloc[test_idx] = proba
        covered.iloc[test_idx] = True

    return pd.DataFrame(fold_scores), oof_proba[covered], df[TARGET][covered]


if __name__ == '__main__':
    df = pd.read_csv('data/processed/model_data.csv', parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    params = load_best_params()
    print('Best params from Optuna study:', params)

    fold_scores, y_proba, y_true = evaluate_tuned_xgb(df, params)
    fold_scores.to_csv('outputs/tables/xgboost_tuned_by_fold.csv', index=False)

    summary = fold_scores[['pr_auc', 'roc_auc']].agg(['mean', 'std']).round(4)
    print('\n=== xgboost_tuned summary across folds ===')
    print(summary)

    # cost-based threshold (raw)
    best_raw, table = find_best_threshold(y_true, y_proba)
    naive_cost, naive_fn, naive_fp = total_cost(y_true, y_proba, 0.5)
    savings_per_1000 = (naive_cost - best_raw['total_cost']) / len(y_true) * 1000
    print(f"\nNaive 0.5 cutoff -> cost={naive_cost:,.0f} (FN={naive_fn}, FP={naive_fp})")
    print(f"Best RAW threshold={best_raw['threshold']:.3f} -> cost={best_raw['total_cost']:,.0f} "
          f"(FN={best_raw['fn']:.0f}, FP={best_raw['fp']:.0f})")
    print(f"Savings per 1,000 (raw threshold): KES {savings_per_1000:,.0f}")

    # calibration
    brier_raw = brier_score_loss(y_true, y_proba)
    iso_proba = isotonic_calibrate(y_true, y_proba)
    brier_iso = brier_score_loss(y_true, iso_proba)
    best_iso, _ = find_best_threshold(y_true, iso_proba)
    savings_iso = (naive_cost - best_iso['total_cost']) / len(y_true) * 1000

    print(f"\nBrier (raw): {brier_raw:.5f} | Brier (isotonic): {brier_iso:.5f}")
    print(f"Best ISOTONIC threshold={best_iso['threshold']:.3f} -> cost={best_iso['total_cost']:,.0f} "
          f"(FN={best_iso['fn']:.0f}, FP={best_iso['fp']:.0f})")
    print(f"Savings per 1,000 (calibrated threshold): KES {savings_iso:,.0f}")

    pd.DataFrame([{
        'model': 'xgboost_tuned',
        'pr_auc_mean': summary.loc['mean', 'pr_auc'], 'pr_auc_std': summary.loc['std', 'pr_auc'],
        'roc_auc_mean': summary.loc['mean', 'roc_auc'], 'roc_auc_std': summary.loc['std', 'roc_auc'],
        'naive_cost': naive_cost, 'naive_fn': naive_fn, 'naive_fp': naive_fp,
        'best_threshold_raw': best_raw['threshold'], 'best_cost_raw': best_raw['total_cost'],
        'savings_per_1000_raw': savings_per_1000,
        'brier_raw': brier_raw, 'brier_isotonic': brier_iso,
        'best_threshold_isotonic': best_iso['threshold'], 'best_cost_isotonic': best_iso['total_cost'],
        'savings_per_1000_isotonic': savings_iso,
    }]).to_csv('outputs/tables/xgboost_tuned_final_summary.csv', index=False)

    y_proba.to_csv('outputs/tables/oof_proba_xgboost_tuned.csv')
    iso_proba.to_csv('outputs/tables/oof_proba_xgboost_tuned_calibrated.csv')
