import json
import joblib
import pandas as pd
from config import (
    VECTORIZER_PATH, MATRIX_PATH, METADATA_PATH,
    METRICS_PATH, EVALUATION_DIR
)

def main():
    for path in [VECTORIZER_PATH, MATRIX_PATH, METADATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {path}. Run train.py first."
            )

    vectorizer = joblib.load(VECTORIZER_PATH)
    matrix = joblib.load(MATRIX_PATH)
    schemes = pd.read_csv(METADATA_PATH)

    summary = {
        "scheme_count": int(len(schemes)),
        "matrix_rows": int(matrix.shape[0]),
        "vector_dimensions": int(matrix.shape[1]),
        "vocabulary_size": int(
            len(vectorizer.get_feature_names_out())
        ),
        "index_consistent": len(schemes) == matrix.shape[0],
        "warning": (
            "Text similarity does not verify legal eligibility. "
            "Eligibility rules must be checked separately."
        )
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
