import argparse
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import (
    VECTORIZER_PATH, MATRIX_PATH, METADATA_PATH,
    ARTIFACTS_DIR, ID_COLUMN, NAME_COLUMN
)
from preprocessing import prepare_worker_profile

def main():
    parser = argparse.ArgumentParser(
        description="Find potentially relevant welfare schemes."
    )
    parser.add_argument("--occupation", default="")
    parser.add_argument("--state", required=True)
    parser.add_argument("--age", default="")
    parser.add_argument("--income", default="")
    parser.add_argument("--worker-category", default="")
    parser.add_argument("--needs", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    for path in [VECTORIZER_PATH, MATRIX_PATH, METADATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {path}. Run train.py first."
            )

    vectorizer = joblib.load(VECTORIZER_PATH)
    matrix = joblib.load(MATRIX_PATH)
    schemes = pd.read_csv(METADATA_PATH)

    profile = prepare_worker_profile(
        args.occupation, args.state, args.age,
        args.income, args.worker_category, args.needs
    )
    profile_vector = vectorizer.transform([profile])
    scores = cosine_similarity(profile_vector, matrix).ravel()

    results = schemes.copy()
    results["relevance_score"] = scores
    results = results.sort_values(
        "relevance_score", ascending=False
    ).head(args.top_k)

    columns = [
        c for c in [
            ID_COLUMN, NAME_COLUMN, "description",
            "eligibility", "benefits", "relevance_score"
        ] if c in results.columns
    ]
    print(results[columns].to_string(index=False))

    output = ARTIFACTS_DIR / "scheme_recommendations.csv"
    results.to_csv(output, index=False)
    print("\nRecommendations saved:", output)
    print(
        "Important: verify official eligibility rules before "
        "presenting any scheme as applicable."
    )

if __name__ == "__main__":
    main()
