import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import (
    RAW_DATA_PATH, ARTIFACTS_DIR, VECTORIZER_PATH,
    MATRIX_PATH, METADATA_PATH, NAME_COLUMN,
    DESCRIPTION_COLUMN, ELIGIBILITY_COLUMN, BENEFITS_COLUMN
)
from preprocessing import prepare_schemes

def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    schemes = prepare_schemes(
        df, NAME_COLUMN, DESCRIPTION_COLUMN,
        ELIGIBILITY_COLUMN, BENEFITS_COLUMN
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=20000
    )
    matrix = vectorizer.fit_transform(schemes["scheme_text"])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(matrix, MATRIX_PATH)
    schemes.drop(columns=["scheme_text"]).to_csv(
        METADATA_PATH, index=False
    )

    print("Scheme vectorizer saved:", VECTORIZER_PATH)
    print("Scheme matrix saved:", MATRIX_PATH)
    print("Schemes indexed:", len(schemes))

if __name__ == "__main__":
    main()
