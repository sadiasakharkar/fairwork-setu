import json
import joblib
import pandas as pd
from config import (
    ARTIFACTS_DIR, MODEL_PATH, COURSE_MATRIX_PATH,
    COURSE_METADATA_PATH, METRICS_PATH, EVALUATION_DIR
)

def main():
    paths = [
        MODEL_PATH, COURSE_MATRIX_PATH, COURSE_METADATA_PATH
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {path}. Run train.py first."
            )

    vectorizer = joblib.load(MODEL_PATH)
    matrix = joblib.load(COURSE_MATRIX_PATH)
    courses = pd.read_csv(COURSE_METADATA_PATH)

    summary = {
        "course_count": int(len(courses)),
        "matrix_rows": int(matrix.shape[0]),
        "matrix_features": int(matrix.shape[1]),
        "vocabulary_size": int(
            len(vectorizer.get_feature_names_out())
        ),
        "index_consistent": len(courses) == matrix.shape[0],
        "evaluation_note": (
            "These are integrity checks, not recommendation-quality metrics. "
            "Validate relevance with learner feedback or labeled interactions."
        )
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
