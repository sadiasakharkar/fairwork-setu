import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import (
    RAW_DATA_PATH, ARTIFACTS_DIR, MODEL_PATH,
    COURSE_MATRIX_PATH, COURSE_METADATA_PATH,
    TITLE_COLUMN, DESCRIPTION_COLUMN, SKILLS_COLUMN
)
from preprocessing import prepare_courses

def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    courses = prepare_courses(
        df, TITLE_COLUMN, DESCRIPTION_COLUMN, SKILLS_COLUMN
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )
    matrix = vectorizer.fit_transform(courses["course_text"])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODEL_PATH)
    joblib.dump(matrix, COURSE_MATRIX_PATH)
    courses.drop(columns=["course_text"]).to_csv(
        COURSE_METADATA_PATH, index=False
    )

    print("Skill vectorizer saved:", MODEL_PATH)
    print("Course matrix saved:", COURSE_MATRIX_PATH)
    print("Courses indexed:", len(courses))

if __name__ == "__main__":
    main()
