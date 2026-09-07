"""
Explainability (SHAP, global + local) and subgroup fairness for the final
selected model (tuned XGBoost, threshold = 0.08 on isotonic-calibrated scores).

Subgroups used (adapted from the brief's 'tenure band or region' since this
dataset has no loan-tenure field):
  - account_age_days, binned into bands  -> tenure proxy (days since a
    customer's first observed transaction)
  - region (already present)

Metrics reported per subgroup, using the chosen threshold:
  - "approval rate"      = share of transactions NOT flagged as fraud (let through)
  - "false-negative rate" = share of ACTUAL fraud cases in that subgroup that were missed
"""
import ast
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from cv import purged_blocked_time_series_split
from models import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, TIME_COL
from calibration import isotonic_calibrate
from final_model import load_best_params

FINAL_THRESHOLD = 0.08  # from final_model.py - isotonic-calibrated cost-optimal cutoff


def train_final_model_on_all_but_last_fold(df, params):
    """Train the deployed model the same way it would be trained in production:
    on the most recent large training window (all data before the final test
    fold), so SHAP explanations reflect the actual deployed model's behaviour."""
    folds = list(purged_blocked_time_series_split(df[TIME_COL], n_splits=5, purge_days=30))
    train_idx, test_idx = folds[-1]
    X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = X_train[TARGET], X_test[TARGET]
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    prep = build_preprocessor()
    Xtr = prep.fit_transform(X_train)
    Xte = prep.transform(X_test)
    feature_names = (NUMERIC_FEATURES +
                      list(prep.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)))

    model = xgb.XGBClassifier(**params, scale_pos_weight=pos_weight,
                               eval_metric='aucpr', random_state=42, n_jobs=-1)
    model.fit(Xtr, y_train)
    return model, Xtr, Xte, X_test.reset_index(drop=True), y_test.reset_index(drop=True), feature_names


def run_shap(model, Xte, feature_names, out_prefix='outputs/figures/shap'):
    sample_n = min(2000, Xte.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(Xte.shape[0], sample_n, replace=False)
    Xte_sample = Xte[sample_idx] if not hasattr(Xte, 'iloc') else Xte.iloc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(Xte_sample)
    shap_values.feature_names = feature_names

    # global explanation
    plt.figure()
    shap.summary_plot(shap_values, Xte_sample, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_global_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # local explanation for one flagged high-risk transaction
    proba = model.predict_proba(Xte_sample)[:, 1]
    highest_risk_idx = int(np.argmax(proba))
    plt.figure()
    shap.plots.waterfall(shap_values[highest_risk_idx], show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_local_example.png', dpi=150, bbox_inches='tight')
    plt.close()

    return sample_idx, proba, highest_risk_idx


def subgroup_metrics(X_test, y_test, y_pred_flag, group_col, bins=None, labels=None):
    df = X_test.copy()
    df['y_true'] = y_test.values
    df['flagged'] = y_pred_flag

    if bins is not None:
        df[group_col + '_band'] = pd.cut(df[group_col], bins=bins, labels=labels)
        group_key = group_col + '_band'
    else:
        group_key = group_col

    rows = []
    for g, sub in df.groupby(group_key, observed=True):
        n = len(sub)
        n_actual_fraud = sub['y_true'].sum()
        approval_rate = 1 - sub['flagged'].mean()  # not flagged = "approved"/let through
        if n_actual_fraud > 0:
            fn = ((sub['y_true'] == 1) & (sub['flagged'] == 0)).sum()
            fnr = fn / n_actual_fraud
        else:
            fnr = np.nan
        rows.append({'group': g, 'n': n, 'n_actual_fraud': n_actual_fraud,
                      'approval_rate': approval_rate, 'false_negative_rate': fnr})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = pd.read_csv('data/processed/model_data.csv', parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    params = load_best_params()

    model, Xtr, Xte, X_test, y_test, feature_names = train_final_model_on_all_but_last_fold(df, params)

    print('Running SHAP (global + local)...')
    sample_idx, proba_sample, highest_risk_idx = run_shap(model, Xte, feature_names)
    print('Saved outputs/figures/shap_global_summary.png and shap_local_example.png')

    # full-test-fold predictions for fairness (not just the SHAP sample)
    proba_full = model.predict_proba(Xte)[:, 1]
    calibrated_full = isotonic_calibrate(y_test, pd.Series(proba_full)).values
    flagged = (calibrated_full >= FINAL_THRESHOLD).astype(int)

    print(f'\nOverall flag rate at threshold {FINAL_THRESHOLD}: {flagged.mean():.3f}')
    print(f'Overall false negative rate: '
          f'{((y_test==1) & (flagged==0)).sum() / max((y_test==1).sum(),1):.3f}')

    age_bins = [-1, 30, 90, 365, 1e6]
    age_labels = ['<30d (new)', '30-90d', '90-365d', '>365d (established)']
    age_table = subgroup_metrics(X_test, y_test, flagged, 'account_age_days', bins=age_bins, labels=age_labels)
    print('\n=== Fairness by account age (tenure proxy) ===')
    print(age_table)

    region_table = subgroup_metrics(X_test, y_test, flagged, 'region')
    print('\n=== Fairness by region ===')
    print(region_table)

    age_table.to_csv('outputs/tables/fairness_by_tenure.csv', index=False)
    region_table.to_csv('outputs/tables/fairness_by_region.csv', index=False)
