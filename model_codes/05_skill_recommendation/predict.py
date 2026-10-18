import argparse
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import (
    ARTIFACTS_DIR, MODEL_PATH, COURSE_MATRIX_PATH,
    COURSE_METADATA_PATH
)
from preprocessing import prepare_skill_profile

def main():
    parser = argparse.ArgumentParser(
        description="Recommend courses based on a worker's skills."
    )
    parser.add_argument("--skills", required=True,
                        help="Current skills, separated by commas")
    parser.add_argument("--role", default="",
                        help="Target job or role")
    parser.add_argument("--interests", default="",
                        help="Topics the learner wants to explore")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    for path in [MODEL_PATH, COURSE_MATRIX_PATH, COURSE_METADATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {path}. Run train.py first."
            )

    vectorizer = joblib.load(MODEL_PATH)
    matrix = joblib.load(COURSE_MATRIX_PATH)
    courses = pd.read_csv(COURSE_METADATA_PATH)

    profile = prepare_skill_profile(
        args.skills, args.role, args.interests
    )
    profile_vector = vectorizer.transform([profile])
    scores = cosine_similarity(profile_vector, matrix).ravel()

    results = courses.copy()
    results["relevance_score"] = scores
    results = results.sort_values(
        "relevance_score", ascending=False
    ).head(args.top_k)

    display_columns = [
        c for c in [
            "course_id", "title", "skills", "level", "relevance_score"
        ] if c in results.columns
    ]
    print(results[display_columns].to_string(index=False))

    output_path = ARTIFACTS_DIR / "skill_recommendations.csv"
    results.to_csv(output_path, index=False)
    print("\nRecommendations saved:", output_path)
    print("Scores indicate text similarity, not guaranteed learning outcomes.")

if __name__ == "__main__":
    main()
