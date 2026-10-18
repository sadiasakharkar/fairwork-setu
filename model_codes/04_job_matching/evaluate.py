import json
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import (
    ARTIFACTS_DIR, MODEL_PATH, JOB_VECTORIZER_PATH,
    EVALUATION_DIR, EVALUATION_PATH
)

def main():
    metadata_path = ARTIFACTS_DIR / "job_metadata.csv"
    ids_path = ARTIFACTS_DIR / "job_ids.txt"

    for path in [MODEL_PATH, JOB_VECTORIZER_PATH,
                 metadata_path, ids_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {path}. Run train.py first."
            )

    vectorizer = joblib.load(JOB_VECTORIZER_PATH)
    job_matrix = joblib.load(MODEL_PATH)
    metadata = pd.read_csv(metadata_path)
    job_ids = ids_path.read_text(encoding="utf-8").splitlines()

    # Basic artifact/index integrity check, not a relevance benchmark.
    summary = {
        "indexed_jobs": int(job_matrix.shape[0]),
        "vector_dimensions": int(job_matrix.shape[1]),
        "metadata_rows": int(len(metadata)),
        "job_ids": int(len(job_ids)),
        "index_consistent": (
            job_matrix.shape[0] == len(metadata) == len(job_ids)
        ),
        "evaluation_note": (
            "Offline relevance quality is not measured here. "
            "Use labeled user-job interactions or human review."
        )
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
