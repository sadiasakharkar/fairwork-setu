from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "transactions.csv"

DATE_COLUMN = "date"
AMOUNT_COLUMN = "amount"
TYPE_COLUMN = "type"

INCOME_VALUES = ["income", "earning", "credit"]
EXPENSE_VALUES = ["expense", "spending", "debit"]

EVALUATION_DIR = BASE_DIR / "evaluation"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
