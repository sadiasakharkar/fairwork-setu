import argparse
import pandas as pd
from config import OUTPUT_DIR
from preprocessing import prepare_transactions, summarize_monthly

def main():
    parser = argparse.ArgumentParser(
        description="Summarize income, expenses and savings."
    )
    parser.add_argument("--input", required=True,
                        help="Path to a transaction CSV")
    parser.add_argument("--output", default="financial_summary.csv",
                        help="Path for monthly summary CSV")
    args = parser.parse_args()

    transactions = prepare_transactions(pd.read_csv(args.input))
    monthly = summarize_monthly(transactions)

    monthly.to_csv(args.output, index=False)
    print(monthly.to_string(index=False))
    print("\nSummary saved:", args.output)
    print(
        "Check transaction classifications and source data before "
        "using these figures for financial decisions."
    )

if __name__ == "__main__":
    main()
