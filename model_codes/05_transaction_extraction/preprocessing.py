import re
import pandas as pd

AMOUNT_PATTERN = re.compile(
    r"(?i)(?:?|rs\.?|inr)\s*([0-9,]+(?:\.[0-9]{1,2})?)"
    r"|([0-9,]+(?:\.[0-9]{1,2})?)\s*(?:rupees?|rs\.?|inr)"
)

DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
]

INCOME_KEYWORDS = [
    "received", "salary", "earned", "credited", "income",
    "payment received", "got paid", "wages"
]
EXPENSE_KEYWORDS = [
    "spent", "paid", "bought", "purchase", "debit",
    "expense", "bill", "recharged"
]

CATEGORY_KEYWORDS = {
    "food": ["food", "grocery", "groceries", "restaurant", "vegetables"],
    "transport": ["bus", "train", "taxi", "auto", "fuel", "petrol", "travel"],
    "health": ["medicine", "hospital", "doctor", "medical"],
    "education": ["school", "college", "course", "tuition"],
    "housing": ["rent", "electricity", "water bill"],
    "salary": ["salary", "wages", "payroll"],
    "shopping": ["clothes", "shopping", "purchase"],
}

def extract_amount(text):
    match = AMOUNT_PATTERN.search(text)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None

def extract_date(text):
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            parsed = pd.to_datetime(match.group(0), errors="coerce", dayfirst=True)
            if not pd.isna(parsed):
                return parsed.strftime("%Y-%m-%d")
    return None

def extract_type(text):
    lowered = text.lower()

    income_hits = sum(word in lowered for word in INCOME_KEYWORDS)
    expense_hits = sum(word in lowered for word in EXPENSE_KEYWORDS)

    if income_hits > expense_hits and income_hits > 0:
        return "income"
    if expense_hits > income_hits and expense_hits > 0:
        return "expense"
    return "unknown"

def extract_category(text):
    lowered = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(word in lowered for word in keywords):
            return category
    return "other"

def extract_transaction(text):
    text = str(text).strip()
    return {
        "amount": extract_amount(text),
        "transaction_type": extract_type(text),
        "category": extract_category(text),
        "date": extract_date(text)
    }

def prepare_dataset(df, text_column):
    if text_column not in df.columns:
        raise ValueError(
            f"Missing text column '{text_column}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    data = df.copy()
    data[text_column] = data[text_column].fillna("").astype(str).str.strip()
    data = data[data[text_column] != ""].copy()

    if data.empty:
        raise ValueError("No non-empty transaction text records found.")

    extracted = data[text_column].apply(extract_transaction).apply(pd.Series)
    return pd.concat([data.reset_index(drop=True), extracted], axis=1)
