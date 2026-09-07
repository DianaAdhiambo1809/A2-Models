"""
Optuna study for the XGBoost model (the strongest candidate for further tuning
since it's the one deployed with early stopping / regularisation controls).

Budget-limited: 60 trials, each trial evaluated on 3 of the 5 purged/blocked
time-series folds (folds 3-5, the ones with the most stable training history)
to keep runtime manageable while still measuring generalisation across
different time periods, not just one split.

Pruning rule: MedianPruner - after each fold's score is reported, a trial is
stopped early if its running average is worse than the median of other trials
at the same step. This avoids wasting time on clearly bad hyperparameter
combinations once a handful of trials have completed.
"""
import pandas as pd
import numpy as np
import optuna
import os
import time
from optuna.pruners import MedianPruner
from sklearn.metrics import average_precision_score
import xgboost as xgb

from cv import purged_blocked_time_series_split
from models import build_preprocessor, TARGET, TIME_COL

# Run in smaller chunks across multiple process invocations (sandbox does not
# keep background processes alive between tool calls); the SQLite study storage
# below lets each invocation resume the same study rather than starting over.
N_TRIALS = int(os.environ.get('N_TRIALS', 15))
TIME_BUDGET_SECONDS = int(os.environ.get('TIME_BUDGET_SECONDS', 240))
FOLDS_FOR_TUNING = [2, 3, 4]  # 0-indexed -> folds 3, 4, 5 of the 5 purged folds


def get_tuning_folds(df):
    all_folds = list(purged_blocked_time_series_split(df[TIME_COL], n_splits=5, purge_days=30))
    return [all_folds[i] for i in FOLDS_FOR_TUNING]


def objective(trial, df, folds, cached_transforms):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }

    scores = []
    for step, (train_idx, test_idx) in enumerate(folds):
        Xtr, ytr, Xte, yte, pos_weight = cached_transforms[step]
        model = xgb.XGBClassifier(**params, scale_pos_weight=pos_weight,
                                   eval_metric='aucpr', random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        score = average_precision_score(yte, proba)
        scores.append(score)

        trial.report(np.mean(scores), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


def precompute_fold_transforms(df, folds):
    """Fit the preprocessor once per fold (not once per trial) - the preprocessing
    itself has no tunable hyperparameters, so refitting it 60 times per fold would
    just waste time."""
    cached = []
    for train_idx, test_idx in folds:
        X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = X_train[TARGET], X_test[TARGET]
        prep = build_preprocessor()
        Xtr = prep.fit_transform(X_train)
        Xte = prep.transform(X_test)
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        cached.append((Xtr, y_train.values, Xte, y_test.values, pos_weight))
    return cached


if __name__ == '__main__':
    df = pd.read_csv('data/processed/model_data.csv', parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    folds = get_tuning_folds(df)
    print(f'Tuning on {len(folds)} folds, fold sizes:', [(len(tr), len(te)) for tr, te in folds])

    cached_transforms = precompute_fold_transforms(df, folds)

    study = optuna.create_study(
        study_name='xgboost_fraud_tuning',
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage='sqlite:///outputs/optuna_study.db',
        load_if_exists=True,
    )
    t0 = time.time()
    study.optimize(lambda t: objective(t, df, folds, cached_transforms),
                    n_trials=N_TRIALS, timeout=TIME_BUDGET_SECONDS)
    elapsed = time.time() - t0

    n_done = len(study.trials)
    print(f'\nThis run: requested up to {N_TRIALS} trials, ran for {elapsed:.0f}s, '
          f'study now has {n_done} total trials.')
    print('=== Best trial so far ===')
    print('Value (mean PR-AUC across tuning folds):', study.best_value)
    print('Params:', study.best_params)

    # Save optimization history + param importance for the report (safe to
    # regenerate every run - reflects the study's cumulative state so far)
    history = study.trials_dataframe()
    history.to_csv('outputs/tables/optuna_trials.csv', index=False)

    with open('outputs/tables/best_xgb_params.txt', 'w') as f:
        f.write(str(study.best_params))

    if n_done >= 60:
        import matplotlib
        matplotlib.use('Agg')
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        fig1 = plot_optimization_history(study)
        fig1.figure.savefig('outputs/figures/optuna_history.png', dpi=150, bbox_inches='tight')
        fig2 = plot_param_importances(study)
        fig2.figure.savefig('outputs/figures/optuna_param_importance.png', dpi=150, bbox_inches='tight')
        print('\n60+ trials complete. Saved optuna_history.png and optuna_param_importance.png')
    else:
        print(f'\n{n_done}/60 trials done - run again to continue (study resumes from SQLite).')
