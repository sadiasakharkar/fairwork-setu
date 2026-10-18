# Model 1: Expense Classification

## Objective
Classify transaction details into expense categories using TF-IDF and Logistic Regression.

## Files
- eda.py: Inspect the dataset and save an EDA summary.
- preprocessing.py: Clean rows and prepare transaction text.
- train.py: Split data, train the model, and save artifacts.
- evaluate.py: Evaluate the trained model on held-out test data.
- predict.py: Predict a category for a new transaction.
- config.py: Configure dataset paths and target column.

## Dataset
Place the CSV file at data/raw/transactions.csv.
Set TARGET_COLUMN in config.py to the actual category-label column.

Each row must have transaction information and a known category label.
Do not include personal or sensitive information unnecessarily.

## Setup
Install dependencies:
pip install -r requirements.txt

## Run in order
python eda.py
python train.py
python evaluate.py
python predict.py --text "Uber ride 250 rupees"

## Outputs
- artifacts/expense_classifier.joblib
- artifacts/metadata.json
- data/processed/test_data.csv
- evaluation/eda_summary.txt
- evaluation/metrics.json

## Notes
Training and testing use separate data splits.
Review class counts and evaluation metrics before using predictions in the app.
