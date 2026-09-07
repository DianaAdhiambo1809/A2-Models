"""
Purged / blocked time-series cross-validation.

Why this design (for the report):
- The data spans Feb 2025 -> Jul 2026 (~18 months), one row per transaction,
  ordered in time. A model that will be deployed to score *future* transactions
  must be evaluated the same way: trained only on the past, tested only on
  the future ("blocked", walk-forward).
- Several engineered features (txn_count_30d, cashout_ratio_30d, etc.) use a
  trailing 30-day rolling window. If a test fold started immediately after a
  training fold ended, its first few rows would have rolling features computed
  partly from data that sits right at the boundary, and a naive split can also
  let information about the *test period's* market conditions leak backward
  during feature recalculation in some pipelines. To be safe, we insert a
  purge/embargo gap of `purge_days` between the end of the training window and
  the start of the test window, wide enough to clear the longest rolling
  window (30 days) used in feature engineering.
"""
import numpy as np
import pandas as pd


def purged_blocked_time_series_split(time_series: pd.Series, n_splits: int = 5, purge_days: int = 30):
    """
    Walk-forward / blocked time-series split with a purge gap.

    Assumes `time_series` is already sorted ascending (caller's responsibility -
    the whole modeling dataset is sorted by time once, upstream).

    Yields (train_idx, test_idx) as positional integer arrays, n_splits times.
    Each test block is a contiguous, non-overlapping slice of time; training
    data is everything strictly before (test_start - purge_days).
    """
    n = len(time_series)
    t = time_series.reset_index(drop=True)
    start, end = t.iloc[0], t.iloc[-1]
    total_days = (end - start).days

    # Reserve the first block purely as "warm-up" training history so the very
    # first test fold still has a meaningful amount of training data.
    block_span = total_days / (n_splits + 1)

    for i in range(1, n_splits + 1):
        test_start = start + pd.Timedelta(days=block_span * i)
        test_end = start + pd.Timedelta(days=block_span * (i + 1))
        purge_cutoff = test_start - pd.Timedelta(days=purge_days)

        train_mask = t < purge_cutoff
        if i == n_splits:
            test_mask = (t >= test_start) & (t <= end)
        else:
            test_mask = (t >= test_start) & (t < test_end)

        train_idx = np.where(train_mask.values)[0]
        test_idx = np.where(test_mask.values)[0]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


if __name__ == '__main__':
    df = pd.read_csv('data/processed/model_data.csv', parse_dates=['txn_time_clean'])
    df = df.sort_values('txn_time_clean').reset_index(drop=True)
    for i, (tr, te) in enumerate(purged_blocked_time_series_split(df['txn_time_clean'], n_splits=5, purge_days=30), 1):
        print(f"Fold {i}: train={len(tr)} rows ({df['txn_time_clean'].iloc[tr].min()} -> {df['txn_time_clean'].iloc[tr].max()}), "
              f"test={len(te)} rows ({df['txn_time_clean'].iloc[te].min()} -> {df['txn_time_clean'].iloc[te].max()}), "
              f"test fraud rate={df['is_fraud'].iloc[te].mean():.4f}")
