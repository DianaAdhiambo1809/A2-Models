# a2-models

DSA 8401, MSc Data Science & Analytics, Assignment 2: The Cost of Being Wrong.

The full write-up is in [`Report/A2_Report.pdf`](Report/A2_Report.pdf). This README is just a guide to the code behind it.

## What this project actually does

The goal is to build a model that scores a mobile money transaction for fraud risk, then pick the exact point at which that score is treated as fraud, based on what each type of mistake actually costs rather than a default 50 percent cutoff. Everything in `src` builds toward that, from cleaning the data through to a final, reloadable model.

One thing worth explaining upfront. The assignment brief is written around a loan default scenario, borrowers, missed defaults, the CBK Digital Credit Providers Regulations. The dataset we actually have from Assignment 1 is mobile money transaction fraud, not lending. We raised this with the course team before submitting, and rather than force the numbers into a story that doesn't fit, we kept the cost logic the assignment asks for and applied it to the problem we actually have. A missed fraud case stands in for a missed default, a wrongly flagged transaction stands in for a wrongly rejected borrower, and so on. The full reasoning behind that mapping is in the report's introduction, not repeated here.

## How the repo is organised

```
a2-models/
├── data/
│   ├── raw/mobile_money_statements.csv     original Assignment 1 input, untouched
│   └── processed/model_data.csv            output of src/preprocess.py
├── src/
│   ├── preprocess.py       reproduces Assignment 1's feature engineering, leaves out the two known leakage columns
│   ├── cv.py                purged, blocked time-series split with a 30 day gap
│   ├── models.py             trains and compares logistic regression, random forest, XGBoost
│   ├── cost.py                searches for the cost-minimising decision threshold
│   ├── imbalance.py            compares class weighting against SMOTE, including a leakage demo
│   ├── tuning.py                 Optuna search over XGBoost's settings, saved to SQLite
│   ├── calibration.py             checks and fixes how trustworthy the model's probabilities are
│   ├── final_model.py              the tuned model evaluated properly, with cost and calibration applied
│   ├── fairness.py                  SHAP explanations and a fairness check across regions
│   └── final_pipeline.py             builds the one file that does preprocessing and prediction together
├── outputs/
│   ├── figures/             every chart used in the report
│   ├── tables/               every results table as csv
│   ├── optuna_study.db        the full tuning history
│   ├── final_fraud_pipeline.joblib   the saved, ready to reload model
│   └── final_pipeline_metadata.json   what threshold and settings it uses
├── Report/
│   ├── A2_Report.pdf          the final report
│   └── latex/main.tex          its LaTeX source
├── requirements.txt
└── README.md
```

## Running it yourself

Each script in `src` picks up where the last one left off, so they're meant to be run in this order.

```bash
pip install -r requirements.txt

cd src
python preprocess.py       # turns the raw csv into the cleaned, feature-engineered dataset
python models.py           # trains the three baseline models and compares them
python cost.py             # finds the cost-optimal decision threshold
python calibration.py      # checks calibration and recalculates the threshold on fixed probabilities
python imbalance.py        # compares imbalance-handling methods, including the leakage example
python tuning.py           # runs the Optuna search, safe to rerun if it gets interrupted partway
python final_model.py      # evaluates the tuned model properly
python fairness.py         # generates the SHAP and fairness results
python final_pipeline.py   # builds and tests the final saved model

cd ../Report/latex
pdflatex main.tex          # rebuilds the report PDF from source
cp main.pdf ../A2_Report.pdf
```

Everything is seeded with `random_state=42`, so rerunning it should give you the same numbers.

## What it actually found

The model that ended up performing best was a tuned XGBoost, PR-AUC of 0.1191, and it was also the most consistent across the five time-based folds, which mattered more to us than a slightly higher score with more swing to it. It runs at a decision threshold of 0.08 on calibrated probabilities, which lines up closely with the 0.074 you'd get from the cost ratio directly, that agreement is a big part of why we trust the number.

Two other things worth knowing before opening the code. First, we deliberately broke our own methodology once to prove a point, resampling the data before splitting it into train and test gave a PR-AUC of 1.00, which looks perfect and is actually just leakage, done correctly the same comparison gives about 0.10. Second, the model doesn't perform equally well everywhere, it misses real fraud in Arusha at 51.4 percent versus 36.8 percent in Mwanza, and that gap is flagged in the report as something to investigate before this goes anywhere near production.
