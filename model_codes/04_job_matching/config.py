from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "jobs.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EVALUATION_DIR = BASE_DIR / "evaluation"

MODEL_PATH = ARTIFACTS_DIR / "job_matcher.joblib"
JOB_VECTORIZER_PATH = ARTIFACTS_DIR / "job_vectorizer.joblib"
USER_VECTORIZER_PATH = ARTIFACTS_DIR / "user_vectorizer.joblib"
EVALUATION_PATH = EVALUATION_DIR / "evaluation_summary.json"

RANDOM_STATE = 42
TOP_K = 10

# Adapt these column names after the finalized dataset is available.
JOB_ID_COLUMN = "job_id"
TITLE_COLUMN = "title"
DESCRIPTION_COLUMN = "description"
SKILLS_COLUMN = "skills"
LOCATION_COLUMN = "location"
WAGE_COLUMN = "wage"
