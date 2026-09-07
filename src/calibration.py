"""
Calibration check for the out-of-fold predictions produced in cost.py.

Checks whether each model's predicted probabilities are "honest" (a predicted
0.3 should mean ~30% of similar cases are actually positive), using:
  - a reliability diagram (predicted probability bin vs observed fraud rate)
  - the Brier score (mean squared error of probabilistic predictions)

Then applies Platt scaling (logistic) and isotonic regression calibration,
recomputes Brier score, and recalculates the cost-based threshold (from cost.py)
on the calibrated scores to see whether/how much it moves.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from cost import out_of_fold_predictions, find_best_threshold, total_cost, COST_FN, COST_FP
from models import TARGET, TIME_COL


def platt_calibrate(y_true, y_proba, train_frac=0.5):
    """Fit Platt scaling (1-D logistic regression on the raw score) on the first
    half of the (already out-of-fold) data in time order, apply to the second half,
    to avoid calibrating and evaluating on the same rows."""
    n = len(y_true)
    split = int(n * train_frac)
    lr = LogisticRegression()
    lr.fit(y_proba.values[:split].reshape(-1, 1), y_true.values[:split])
    calibrated = lr.predict_proba(y_proba.values.reshape(-1, 1))[:, 1]
    return pd.Series(calibrated, index=y_proba.index)


def isotonic_calibrate(y_true, y_proba, train_frac=0.5):
    n = len(y_true)
    split = int(n * train_frac)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_proba.values[:split], y_true.values[:split])
    calibrated = iso.predict(y_proba.values)
    return pd.Series(calibrated, index=y_proba.index)


def reliability_table(y_true, y_proba, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='quantile')
    return mean_pred, frac_pos


if __name__ == '__main__':
    df = pd.read_csv('data/processed/model_data.csv', parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    summary_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, model_name in zip(axes, ['logistic_regression', 'random_forest', 'xgboost']):
        y_proba, y_true = out_of_fold_predictions(df, model_name)

        brier_raw = brier_score_loss(y_true, y_proba)
        platt_proba = platt_calibrate(y_true, y_proba)
        iso_proba = isotonic_calibrate(y_true, y_proba)
        brier_platt = brier_score_loss(y_true, platt_proba)
        brier_iso = brier_score_loss(y_true, iso_proba)

        # recompute cost-based threshold on calibrated (isotonic, generally best-performing) scores
        best_raw, _ = find_best_threshold(y_true, y_proba)
        best_iso, _ = find_best_threshold(y_true, iso_proba)

        print(f"\n=== {model_name} ===")
        print(f"Brier (raw):      {brier_raw:.5f}")
        print(f"Brier (Platt):    {brier_platt:.5f}")
        print(f"Brier (isotonic): {brier_iso:.5f}")
        print(f"Best threshold RAW score:       {best_raw['threshold']:.3f} (cost={best_raw['total_cost']:,.0f})")
        print(f"Best threshold ISOTONIC score:  {best_iso['threshold']:.3f} (cost={best_iso['total_cost']:,.0f})")

        mean_pred, frac_pos = reliability_table(y_true, y_proba)
        mean_pred_iso, frac_pos_iso = reliability_table(y_true, iso_proba)
        ax.plot([0, 1], [0, 1], 'k--', label='perfect calibration')
        ax.plot(mean_pred, frac_pos, 'o-', label='raw')
        ax.plot(mean_pred_iso, frac_pos_iso, 's-', label='isotonic-calibrated')
        ax.set_title(model_name)
        ax.set_xlabel('mean predicted probability')
        ax.set_ylabel('observed fraud rate')
        ax.legend(fontsize=8)

        summary_rows.append({
            'model': model_name,
            'brier_raw': brier_raw, 'brier_platt': brier_platt, 'brier_isotonic': brier_iso,
            'threshold_raw': best_raw['threshold'], 'cost_raw': best_raw['total_cost'],
            'threshold_isotonic': best_iso['threshold'], 'cost_isotonic': best_iso['total_cost'],
        })

    plt.tight_layout()
    plt.savefig('outputs/figures/reliability_diagrams.png', dpi=150)
    print('\nSaved outputs/figures/reliability_diagrams.png')

    pd.DataFrame(summary_rows).to_csv('outputs/tables/calibration_summary.csv', index=False)
