# Model 4: Job Matching & Recommendation

## Purpose
Recommend relevant jobs to informal and gig workers based on skills, experience, role preference, and job text.

## Initial algorithm
TF-IDF text vectorization with cosine similarity. This is a baseline recommendation approach, not a trained hiring-probability model.

## Files
- eda.py: dataset inspection
- preprocessing.py: job text and worker profile preparation
- train.py: build a searchable job-text index
- evaluate.py: index integrity checks
- predict.py: rank jobs for a worker profile
- config.py: paths and expected column names

## Expected dataset
Path: data/raw/jobs.csv

The initial scaffold expects these columns:
- job_id
- title
- description
- skills
- location (optional for display)
- wage (optional for display)

Adapt column names and matching logic after the real dataset is finalized.

## Commands
pip install -r requirements.txt
python eda.py
python train.py
python evaluate.py
python predict.py --skills "driving, delivery, customer service" --role "Delivery Partner" --location "Pune" --top-k 10

## Evaluation
Index consistency checks do not measure recommendation quality. Once user-job interaction data is available, evaluate Precision@K, Recall@K, and NDCG@K where appropriate, and include human relevance review.

## Status
Starter scaffold only. No dataset has been added; the recommender has not been validated.
Antigravity must adapt the code to the finalized dataset and test it before backend integration.
