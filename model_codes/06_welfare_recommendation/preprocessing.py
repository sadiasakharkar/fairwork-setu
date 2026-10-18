import pandas as pd

def prepare_schemes(df, name_column, description_column,
                    eligibility_column, benefits_column):
    required = [
        name_column, description_column,
        eligibility_column, benefits_column
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    schemes = df.copy()
    for column in required:
        schemes[column] = (
            schemes[column].fillna("").astype(str).str.strip()
        )

    schemes["scheme_text"] = (
        schemes[name_column] + " " +
        schemes[description_column] + " " +
        schemes[eligibility_column] + " " +
        schemes[benefits_column]
    ).str.strip()

    schemes = schemes[schemes["scheme_text"] != ""]
    schemes = schemes.drop_duplicates(
        subset=["scheme_text"]
    ).reset_index(drop=True)

    if schemes.empty:
        raise ValueError("No usable welfare scheme records found.")

    return schemes

def prepare_worker_profile(occupation="", state="", age="",
                           income="", worker_category="",
                           needs=""):
    return " ".join([
        str(occupation or ""),
        str(state or ""),
        str(age or ""),
        str(income or ""),
        str(worker_category or ""),
        str(needs or "")
    ]).strip()
