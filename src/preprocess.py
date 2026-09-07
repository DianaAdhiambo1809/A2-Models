"""
Reproduces Assignment 1's data cleaning and feature engineering EXACTLY as built
in the A1 notebook (Mobile_Money_-_Data_Pre-processing_Notebook.ipynb).

Per the A2 brief: "Use the output from Assignment 1, exactly as it was. Don't change it."
This script is a faithful, non-modified translation of that notebook into a
reusable function, so it can be run once to produce a fixed, versioned dataset.

Two columns identified in A1 as intentionally planted leakage sources
(`manual_review_score`, `settlement_status`) are excluded from the modeling
feature set, exactly as A1 concluded.
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime


NUMERIC_FEATURES = [
    'amount_clean', 'balance_after', 'txn_count_7d', 'txn_count_30d',
    'total_amount_30d', 'avg_amount_30d', 'cashout_ratio_30d', 'debit_ratio_30d',
    'hours_since_last_txn', 'is_night_txn', 'day_of_week', 'account_age_days',
    'distinct_devices_30d', 'distinct_agents_30d', 'device_shared_flag',
]
CATEGORICAL_FEATURES = ['segment', 'region']
TARGET = 'is_fraud'
TIME_COL = 'txn_time_clean'
GROUP_COL = 'reg_id_clean'


def _clean_amount(x):
    x = str(x).strip()
    negative = x.startswith('(') and x.endswith(')')
    x = x.strip('()').replace('Dr', '').replace(',', '').replace('/-', '').strip()
    try:
        val = float(x)
    except ValueError:
        return np.nan
    if negative or val < 0:
        return -abs(val)
    return val


def _parse_time(x):
    x = str(x)
    for fmt in ('%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', '%b %d, %Y %I:%M %p'):
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            pass
    return pd.NaT


def load_and_engineer(raw_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)

    # --- identity cleanup (A1 Section 1) ---
    df['reg_id_clean'] = df['reg_id'].str.upper().str.strip()

    # --- amount + time cleanup (A1 Section 1) ---
    df['amount_clean'] = df['amount'].apply(_clean_amount)
    df['direction'] = np.where(df['amount_clean'] < 0, 'debit', 'credit')
    df['txn_time_clean'] = df['txn_time'].apply(_parse_time)

    # --- drop exact duplicate rows, sort by customer + time (A1 Section 1) ---
    df = df.drop_duplicates().copy()
    df = df.sort_values(['reg_id_clean', 'txn_time_clean']).reset_index(drop=True)

    # --- velocity features (A1 Section 2) ---
    df_time = df.set_index('txn_time_clean')
    grouped = df_time.groupby('reg_id_clean')
    df['txn_count_7d'] = grouped['amount_clean'].rolling('7D').count().reset_index(level=0, drop=True).values
    df['txn_count_30d'] = grouped['amount_clean'].rolling('30D').count().reset_index(level=0, drop=True).values
    df['total_amount_30d'] = grouped['amount_clean'].rolling('30D').sum().reset_index(level=0, drop=True).values
    df['avg_amount_30d'] = df['total_amount_30d'] / df['txn_count_30d']

    # --- ratio/behaviour features (A1 Section 2) ---
    df['is_cashout'] = (df['txn_type'] == 'cashout').astype(int)
    df['is_debit'] = (df['direction'] == 'debit').astype(int)
    large_cutoff = df['amount_clean'].abs().quantile(0.75)
    df['is_large_txn'] = (df['amount_clean'].abs() > large_cutoff).astype(int)

    df_time = df.set_index('txn_time_clean')
    grouped = df_time.groupby('reg_id_clean')
    df['cashout_ratio_30d'] = grouped['is_cashout'].rolling('30D').mean().reset_index(level=0, drop=True).values
    df['debit_ratio_30d'] = grouped['is_debit'].rolling('30D').mean().reset_index(level=0, drop=True).values
    df['large_txn_ratio_30d'] = grouped['is_large_txn'].rolling('30D').mean().reset_index(level=0, drop=True).values

    # --- recency features (A1 Section 2) ---
    df['hours_since_last_txn'] = df.groupby('reg_id_clean')['txn_time_clean'].diff().dt.total_seconds() / 3600
    df['is_night_txn'] = df['txn_time_clean'].dt.hour.between(0, 5).astype(int)
    df['day_of_week'] = df['txn_time_clean'].dt.dayofweek
    first_txn = df.groupby('reg_id_clean')['txn_time_clean'].transform('min')
    df['account_age_days'] = (df['txn_time_clean'] - first_txn).dt.total_seconds() / 86400

    # --- device/agent features (A1 Section 2) ---
    device_owner_count = df.groupby('device_id')['reg_id_clean'].nunique()
    shared_devices = device_owner_count[device_owner_count > 1].index
    df['device_shared_flag'] = df['device_id'].isin(shared_devices).astype(int)
    df['device_code'] = pd.factorize(df['device_id'])[0]
    df['agent_code'] = pd.factorize(df['agent_id'])[0]

    df_time = df.set_index('txn_time_clean')
    grouped = df_time.groupby('reg_id_clean')
    df['distinct_devices_30d'] = grouped['device_code'].rolling('30D').apply(lambda x: len(set(x)), raw=True).reset_index(level=0, drop=True).values
    df['distinct_agents_30d'] = grouped['agent_code'].rolling('30D').apply(lambda x: len(set(x)), raw=True).reset_index(level=0, drop=True).values

    keep_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, GROUP_COL, TIME_COL]
    model_df = df[keep_cols].copy()

    # A1 filled engineered-feature NaNs (e.g. first-ever txn has no "since last txn") with 0
    model_df[NUMERIC_FEATURES] = model_df[NUMERIC_FEATURES].fillna(0)

    return model_df


if __name__ == '__main__':
    out = load_and_engineer('data/raw/mobile_money_statements.csv')
    out = out.sort_values(TIME_COL).reset_index(drop=True)
    out.to_csv('data/processed/model_data.csv', index=False)
    print('Shape:', out.shape)
    print('Fraud rate:', out[TARGET].mean().round(4))
    print('Date range:', out[TIME_COL].min(), 'to', out[TIME_COL].max())
    print('Saved to data/processed/model_data.csv')
