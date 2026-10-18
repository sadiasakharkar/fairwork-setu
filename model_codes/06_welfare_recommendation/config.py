from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "schemes.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
EVALUATION_DIR = BASE_DIR / "evaluation"

VECTORIZER_PATH = ARTIFACTS_DIR / "scheme_vectorizer.joblib"
MATRIX_PATH = ARTIFACTS_DIR / "scheme_matrix.joblib"
METADATA_PATH = ARTIFACTS_DIR / "scheme_metadata.csv"
METRICS_PATH = EVALUATION_DIR / "evaluation_summary.json"

ID_COLUMN = "scheme_id"
NAME_COLUMN = "scheme_name"
DESCRIPTION_COLUMN = "description"
ELIGIBILITY_COLUMN = "eligibility"
BENEFITS_COLUMN = "benefits"
