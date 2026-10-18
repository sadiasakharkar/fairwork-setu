import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import (
    RAW_DATA_PATH, ARTIFACTS_DIR, MODEL_PATH,
    JOB_VECTORIZER_PATH, USER_VECTORIZER_PATH,
    JOB_ID_COLUMN, TITLE_COLUMN, DESCRIPTION_COLUMN, SKILLS_COLUMN
)
from preprocessing import prepare_jobs

def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    jobs = prepare_jobs(
        df, TITLE_COLUMN, DESCRIPTION_COLUMN, SKILLS_COLUMN
    )

    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 2), max_features=20000
    )
    job_matrix = vectorizer.fit_transform(jobs["job_text"])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, JOB_VECTORIZER_PATH)
    joblib.dump(job_matrix, MODEL_PATH)

    if JOB_ID_COLUMN in jobs.columns:
        job_ids = jobs[JOB_ID_COLUMN].fillna("").astype(str).tolist()
    else:
        job_ids = jobs.index.astype(str).tolist()

    job_metadata = jobs.drop(columns=["job_text"])
    job_metadata.to_csv(
        ARTIFACTS_DIR / "job_metadata.csv", index=False
    )
    (ARTIFACTS_DIR / "job_ids.txt").write_text(
        "\n".join(job_ids), encoding="utf-8"
    )

    print("Job text index saved:", MODEL_PATH)
    print("Jobs indexed:", len(jobs))
    print("Vector features:", len(vectorizer.get_feature_names_out()))

if __name__ == "__main__":
    main()
