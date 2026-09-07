"""
Cost-based decision threshold.

Business framing (documented explicitly, since our dataset is fraud detection,
not loan lending - see README / report for the full disclosure of this mapping):
    - Missing a real fraud case (false negative)  -> costs KES 10,000
    - Flagging a legitimate transaction as fraud (false positive) -> costs KES 800

We generate OUT-OF-FOLD predictions across the whole purged/blocked timeline
(every row gets a prediction only from a model that never saw it in training),
then sweep thresholds to find the one that minimises total expected cost,
and compare it against the naive 0.5 cutoff.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from cv import purged_blocked_time_series_split
from models import build_preprocessor, TARGET, TIME_COL
from sklearn.pipeline import Pipeline

COST_FN = 10_000   # missed fraud
COST_FP = 800      # legitimate transaction wrongly flagged


def get_model(name, pos_weight):
    if name == 'logistic_regression':
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, random_state=42)
    elif name == 'random_forest':
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced',
                                      random_state=42, n_jobs=-1)
    elif name == 'xgboost':
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 scale_pos_weight=pos_weight, eval_metric='aucpr',
                                 random_state=42, n_jobs=-1)
    else:
        raise ValueError(name)
    return Pipeline([('prep', build_preprocessor()), ('clf', clf)])


def out_of_fold_predictions(df, model_name, n_splits=5, purge_days=30):
    oof_proba = pd.Series(index=df.index, dtype=float)
    covered = pd.Series(False, index=df.index)
    for train_idx, test_idx in purged_blocked_time_series_split(df[TIME_COL], n_splits, purge_days):
        y_train = df.iloc[train_idx][TARGET]
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        pipe = get_model(model_name, pos_weight)
        pipe.fit(df.iloc[train_idx], y_train)
        proba = pipe.predict_proba(df.iloc[test_idx])[:, 1]
        oof_proba.iloc[test_idx] = proba
        covered.iloc[test_idx] = True
    # only the region covered by test folds (i.e. not the very first warm-up window) is usable
    return oof_proba[covered], df[TARGET][covered]


def total_cost(y_true, y_proba, threshold, cost_fn=COST_FN, cost_fp=COST_FP):
    pred = (y_proba >= threshold).astype(int)
    fn = ((pred == 0) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    return fn * cost_fn + fp * cost_fp, fn, fp


def find_best_threshold(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        cost, fn, fp = total_cost(y_true, y_proba, t)
        rows.append({'threshold': t, 'total_cost': cost, 'fn': fn, 'fp': fp})
    table = pd.DataFrame(rows)
    best = table.loc[table['total_cost'].idxmin()]
    return best, table


if __name__ == '__main__':
    df = pd.read_csv('data/processed/model_data.csv', parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    summary_rows = []
    for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
        print(f'\n=== {model_name} ===')
        y_proba, y_true = out_of_fold_predictions(df, model_name)
        y_proba.to_csv(f'outputs/tables/oof_proba_{model_name}.csv')

        best, table = find_best_threshold(y_true, y_proba)
        table.to_csv(f'outputs/tables/threshold_sweep_{model_name}.csv', index=False)

        naive_cost, naive_fn, naive_fp = total_cost(y_true, y_proba, 0.5)
        n = len(y_true)
        savings_per_1000 = (naive_cost - best['total_cost']) / n * 1000

        print(f"n covered by OOF predictions: {n}")
        print(f"Naive 0.5 cutoff   -> cost={naive_cost:,.0f} (FN={naive_fn}, FP={naive_fp})")
        print(f"Best cost cutoff={best['threshold']:.2f} -> cost={best['total_cost']:,.0f} "
              f"(FN={best['fn']:.0f}, FP={best['fp']:.0f})")
        print(f"Savings per 1,000 transactions: KES {savings_per_1000:,.0f}")

        summary_rows.append({
            'model': model_name, 'n': n,
            'naive_cost': naive_cost, 'naive_fn': naive_fn, 'naive_fp': naive_fp,
            'best_threshold': best['threshold'], 'best_cost': best['total_cost'],
            'best_fn': best['fn'], 'best_fp': best['fp'],
            'savings_per_1000': savings_per_1000,
        })

    pd.DataFrame(summary_rows).to_csv('outputs/tables/cost_threshold_summary.csv', index=False)
