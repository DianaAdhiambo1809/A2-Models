# a2-models

DSA 8401 MSc Data Science & Analytics — Assignment 2: *The Cost of Being Wrong*

**Full report:** [`report/A2_Report.pdf`](report/A2_Report.pdf)

## ⚠️ Note on dataset scope (read first)

This assignment's brief is written for a **loan/credit-default** use case (borrowers,
CBK Digital Credit Providers Regulations 2022). Our actual Assignment 1 dataset is
**mobile-money transaction fraud detection** (`is_fraud`, ~8.4% positive rate,
transaction-level). This mismatch was flagged with the course team before submission.
We apply the same cost-sensitive framework to the data we were given, with an explicit
mapping documented in full in the report's Introduction:

| Brief's term | Mapped to |
|---|---|
| Missed default (KES 10,000) | Missed fraudulent transaction |
| Wrongly rejected good borrower (KES 800) | Legitimate transaction wrongly flagged |
| Tenure band | `account_age_days` (days since first observed transaction) |
| Region | used as-is |
| CBK Digital Credit Providers Regulations | CBK fraud/AML/cybersecurity guidance + Data Protection Act 2019 |

## What's in this repo

```
a2-models/
├── data/
│   ├── raw/mobile_money_statements.csv     # original A1 input, unmodified
│   └── processed/model_data.csv            # A1's engineered features, reproduced exactly (src/preprocess.py)
├── src/
│   ├── preprocess.py      # A1 feature engineering, reproduced faithfully (excludes the 2 known leakage columns)
│   ├── cv.py               # purged/blocked time-series CV with a 30-day embargo
│   ├── models.py            # logistic regression / random forest / XGBoost comparison
│   ├── cost.py               # cost-based threshold search (KES 10,000 FN / KES 800 FP)
│   ├── imbalance.py           # class weighting vs SMOTE, + a deliberate leakage demonstration
│   ├── tuning.py                # Optuna study (60+ trials, MedianPruner), SQLite-backed
│   ├── calibration.py            # reliability diagrams, Brier score, threshold recalculation
│   ├── final_model.py             # tuned XGBoost evaluated on full CV + cost + calibration
│   ├── fairness.py                 # SHAP (global+local) + subgroup fairness metrics
│   └── final_pipeline.py            # the single deployable end-to-end sklearn Pipeline
├── outputs/
│   ├── figures/            # all plots referenced in the report
│   ├── tables/              # all CSV result tables
│   ├── optuna_study.db       # full Optuna trial history (SQLite)
│   ├── final_fraud_pipeline.joblib     # the saved, reloadable production pipeline
│   └── final_pipeline_metadata.json     # threshold, params, feature list
├── report/
│   ├── A2_Report.pdf          # the final report (5 pages: cover + 4 content)
│   └── latex/main.tex          # LaTeX source - recompile with pdflatex main.tex
├── requirements.txt
└── README.md
```

## Reproducing this from a fresh clone

```bash
pip install -r requirements.txt

cd src
python preprocess.py          # data/raw -> data/processed/model_data.csv
python models.py                # baseline model comparison table
python cost.py                    # cost-based threshold search
python calibration.py               # reliability diagrams + Brier scores
python imbalance.py                   # class weighting vs SMOTE + leakage demo
python tuning.py                        # Optuna study (resumable via SQLite; run multiple
                                          # times if your machine needs it in smaller chunks -
                                          # it continues from outputs/optuna_study.db)
python final_model.py                     # tuned XGBoost, full evaluation
python fairness.py                          # SHAP + subgroup fairness
python final_pipeline.py                      # builds & reload-tests the deployable pipeline

cd ../report/latex
pdflatex main.tex                 # produces latex/main.pdf
cp main.pdf ../A2_Report.pdf       # (optional) update the committed copy
```

All randomness is seeded (`random_state=42`) for reproducibility.

## Key results (see report for full discussion)

- **Final model:** Optuna-tuned XGBoost — PR-AUC 0.1191 ± 0.0040 (best and most stable of 4 models compared)
- **Decision threshold:** 0.08 on isotonic-calibrated probability (matches the theoretical
  cost-optimal threshold of ≈0.074 for this cost ratio)
- **Evaluation:** purged/blocked time-series CV, 5 folds, 30-day embargo gap
- **Leakage demonstration:** resampling before the train/test split inflated PR-AUC to a
  fabricated 1.00, vs. ~0.10 with correct methodology
- **Fairness finding:** false-negative rate varies from 36.8% (Mwanza) to 51.4% (Arusha)
  across regions — flagged for further investigation before deployment
