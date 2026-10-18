import pandas as pd
from config import RAW_DATA_PATH, OUTPUT_DIR
from preprocessing import prepare_transactions, summarize_monthly

def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    transactions = prepare_transactions(pd.read_csv(RAW_DATA_PATH))
    monthly = summarize_monthly(transactions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_DIR / "monthly_financial_summary.csv", index=False)

    totals = {
        "total_income": float(
            transactions.loc[
                transactions["transaction_group"] == "income", "amount"
            ].sum()
        ),
        "total_expenses": float(
            transactions.loc[
                transactions["transaction_group"] == "expense", "amount"
            ].sum()
        ),
        "unclassified_amount": float(
            transactions.loc[
                transactions["transaction_group"] == "other", "amount"
            ].sum()
        ),
        "transaction_count": int(len(transactions))
    }
    totals["net_savings"] = totals["total_income"] - totals["total_expenses"]

    pd.DataFrame([totals]).to_csv(
        OUTPUT_DIR / "overall_financial_summary.csv", index=False
    )

    print("Overall summary:")
    print(pd.DataFrame([totals]).to_string(index=False))
    print("\nMonthly summary:")
    print(monthly.to_string(index=False))
    print("\nOutputs saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
