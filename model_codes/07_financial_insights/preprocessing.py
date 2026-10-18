import pandas as pd
from config import (
    DATE_COLUMN, AMOUNT_COLUMN, TYPE_COLUMN,
    INCOME_VALUES, EXPENSE_VALUES
)

def prepare_transactions(df):
    required = [DATE_COLUMN, AMOUNT_COLUMN, TYPE_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    data = df[required].copy()
    data[DATE_COLUMN] = pd.to_datetime(
        data[DATE_COLUMN], errors="coerce"
    )
    data[AMOUNT_COLUMN] = pd.to_numeric(
        data[AMOUNT_COLUMN], errors="coerce"
    )
    data[TYPE_COLUMN] = (
        data[TYPE_COLUMN].astype("string").str.strip().str.lower()
    )
    data = data.dropna(subset=required)

    if data.empty:
        raise ValueError("No valid transaction records found.")

    if (data[AMOUNT_COLUMN] < 0).any():
        raise ValueError(
            "Negative amounts found. Confirm the dataset's sign convention."
        )

    income_set = set(INCOME_VALUES)
    expense_set = set(EXPENSE_VALUES)
    data["transaction_group"] = data[TYPE_COLUMN].map(
        lambda value: (
            "income" if value in income_set
            else "expense" if value in expense_set
            else "other"
        )
    )

    return data.sort_values(DATE_COLUMN).reset_index(drop=True)

def summarize_monthly(data):
    data = data.copy()
    data["month"] = data[DATE_COLUMN].dt.to_period("M").astype(str)

    grouped = data.pivot_table(
        index="month",
        columns="transaction_group",
        values=AMOUNT_COLUMN,
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    for column in ["income", "expense", "other"]:
        if column not in grouped.columns:
            grouped[column] = 0.0

    grouped["net_savings"] = grouped["income"] - grouped["expense"]
    grouped["savings_rate_percent"] = (
        grouped["net_savings"] / grouped["income"].replace(0, float("nan"))
    ) * 100

    return grouped
