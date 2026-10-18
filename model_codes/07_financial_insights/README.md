# Model 7: Financial Insights

## Purpose
Summarize income, expenses, net savings and monthly cash flow for Setu's Dhan module.

## Approach
Descriptive analytics using transaction aggregation and monthly summaries. This is not a machine-learning prediction model.

## Files
- eda.py: dataset inspection
- preprocessing.py: transaction cleaning and monthly aggregation
- train.py: generate overall and monthly summaries
- evaluate.py: calculate summary checks and metrics
- predict.py: process a supplied transaction CSV
- config.py: dataset paths and transaction-type labels

## Expected dataset
Path: data/raw/transactions.csv

Expected columns:
- date
- amount
- type

Update column names and income/expense labels after inspecting the finalized dataset. The scaffold assumes amounts are non-negative and transaction type identifies income or expense.

## Commands
pip install -r requirements.txt
python eda.py
python train.py
python evaluate.py
python predict.py --input data/raw/transactions.csv --output financial_summary.csv

## Outputs
- data/processed/monthly_financial_summary.csv
- data/processed/overall_financial_summary.csv
- evaluation/metrics.json

## Status
Starter scaffold only. No dataset has been added; calculations have not been validated against real transaction records.
Antigravity must adapt column names and transaction classifications to the finalized dataset before integration.
