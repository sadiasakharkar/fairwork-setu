import json
import pandas as pd
from config import RAW_DATA_PATH, OUTPUT_DIR, EVALUATION_DIR
from preprocessing import prepare_transactions, summarize_monthly

def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    transactions = prepare_transactions(pd.read_csv(RAW_DATA_PATH))
    monthly = summarize_monthly(transactions)

    income = float(
        transactions.loc[
            transactions["transaction_group"] == "income", "amount"
        ].sum()
    )
    expenses = float(
        transactions.loc[
            transactions["transaction_group"] == "expense", "amount"
        ].sum()
    )
    net = income - expenses

    metrics = {
        "transaction_count": int(len(transactions)),
        "months_covered": int(monthly["month"].nunique()),
        "total_income": income,
        "total_expenses": expenses,
        "net_savings": net,
        "overall_savings_rate_percent": (
            round(net / income * 100, 2) if income else None
        ),
        "unclassified_transaction_count": int(
            (transactions["transaction_group"] == "other").sum()
        ),
        "validation_note": (
            "These are descriptive calculations, not predictive model metrics. "
            "Review transaction types and compare totals against source records."
        )
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    (EVALUATION_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
