from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "courses.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
EVALUATION_DIR = BASE_DIR / "evaluation"

MODEL_PATH = ARTIFACTS_DIR / "skill_vectorizer.joblib"
COURSE_MATRIX_PATH = ARTIFACTS_DIR / "course_matrix.joblib"
COURSE_METADATA_PATH = ARTIFACTS_DIR / "course_metadata.csv"
METRICS_PATH = EVALUATION_DIR / "evaluation_summary.json"

COURSE_ID_COLUMN = "course_id"
TITLE_COLUMN = "title"
DESCRIPTION_COLUMN = "description"
SKILLS_COLUMN = "skills"
LEVEL_COLUMN = "level"
