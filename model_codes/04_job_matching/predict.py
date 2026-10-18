import argparse
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import ARTIFACTS_DIR, MODEL_PATH, JOB_VECTORIZER_PATH
from preprocessing import prepare_user_profile

def main():
    parser = argparse.ArgumentParser(
        description="Recommend jobs matching a worker profile."
    )
    parser.add_argument("--skills", required=True,
                        help="Worker skills, separated by commas")
    parser.add_argument("--experience", default="")
    parser.add_argument("--role", default="")
    parser.add_argument("--location", default="")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    metadata_path = ARTIFACTS_DIR / "job_metadata.csv"
    for path in [MODEL_PATH, JOB_VECTORIZER_PATH, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {path}. Run train.py first."
            )

    vectorizer = joblib.load(JOB_VECTORIZER_PATH)
    job_matrix = joblib.load(MODEL_PATH)
    jobs = pd.read_csv(metadata_path)

    profile = prepare_user_profile(
        args.skills, args.experience, args.role, args.location
    )
    profile_vector = vectorizer.transform([profile])
    scores = cosine_similarity(profile_vector, job_matrix).ravel()

    jobs = jobs.copy()
    jobs["match_score"] = scores
    jobs = jobs.sort_values("match_score", ascending=False)
    results = jobs.head(args.top_k)

    display_columns = [
        c for c in [
            "job_id", "title", "location", "wage", "skills", "match_score"
        ] if c in results.columns
    ]
    if display_columns:
        print(results[display_columns].to_string(index=False))
    else:
        print(results.to_string(index=False))

    output_path = ARTIFACTS_DIR / "latest_recommendations.csv"
    results.to_csv(output_path, index=False)
    print("\nRecommendations saved:", output_path)
    print("Match scores are text-similarity scores, not hiring probabilities.")

if __name__ == "__main__":
    main()
