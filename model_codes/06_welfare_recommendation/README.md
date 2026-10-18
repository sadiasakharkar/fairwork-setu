# Model 6: Welfare Scheme Recommendation

## Purpose
Identify potentially relevant welfare schemes for informal and gig workers based on their profile and stated needs.

## Baseline algorithm
TF-IDF with cosine similarity for ranking scheme descriptions against a worker profile.

## Files
- eda.py: dataset inspection
- preprocessing.py: scheme text and worker profile preparation
- train.py: build a searchable scheme index
- evaluate.py: artifact and index consistency checks
- predict.py: rank potentially relevant schemes
- config.py: paths and expected column names

## Expected dataset
Path: data/raw/schemes.csv

Expected columns:
- scheme_id
- scheme_name
- description
- eligibility
- benefits

Adapt column names to the finalized dataset.

## Commands
pip install -r requirements.txt
python eda.py
python train.py
python evaluate.py
python predict.py --state "Maharashtra" --occupation "delivery worker" --age 28 --income 15000 --worker-category "gig worker" --needs "health insurance and accident support" --top-k 5

## Safety and eligibility
Text similarity is only a discovery aid. It does not establish eligibility, confirm scheme availability, or replace official scheme rules. Validate state, income, age, occupation, documentation, and other required conditions against authoritative sources.

## Evaluation
Index checks do not measure recommendation quality. Use verified scheme-worker examples and expert review to evaluate relevance and false recommendations.

## Status
Starter scaffold only. No dataset has been added; recommendations have not been validated.
Antigravity must adapt and test the code with the finalized dataset before integration.
