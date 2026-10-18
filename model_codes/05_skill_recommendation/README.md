# Model 5: Skill Recommendation & Learning Path

## Purpose
Recommend relevant learning resources to informal and gig workers based on their current skills, target role, and interests.

## Baseline algorithm
TF-IDF vectorization with cosine similarity. This ranks courses by text relevance; it does not prove skill mastery or guarantee career outcomes.

## Files
- eda.py: inspect the course dataset
- preprocessing.py: prepare course text and learner profiles
- train.py: build the searchable course index
- evaluate.py: check artifact and index consistency
- predict.py: recommend relevant courses
- config.py: dataset paths and expected columns

## Expected dataset
Path: data/raw/courses.csv

Expected columns:
- course_id
- title
- description
- skills
- level (optional)

Adapt these names to the finalized dataset.

## Commands
pip install -r requirements.txt
python eda.py
python train.py
python evaluate.py
python predict.py --skills "basic computing, communication" --role "Delivery Partner" --interests "digital payments" --top-k 5

## Evaluation
Integrity checks do not measure recommendation quality. Once learner feedback or course-interaction data is available, assess Precision@K, Recall@K, NDCG@K, and human relevance.

## Status
Starter scaffold only. No dataset has been added; recommendations have not been validated.
Antigravity must adapt the code to the finalized dataset and test it before integration.
