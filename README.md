# E-Commerce Product Recommender with Apache Spark (ALS)

BDA Assignment No. 10, BE Sem-7 Computer Engineering, Vidyalankar Institute of Technology (VIT), Mumbai.

**Authors:** Hasnain Khan (23102A0075) 
**Blog:** [paste published Medium/Hashnode/LinkedIn link]

## Overview
A scalable collaborative-filtering pipeline built with PySpark. It ingests Amazon product ratings, cleans and filters them, trains an ALS matrix-factorization model, compares it with simple baselines, and generates top-10 product recommendations.

## Dataset
Amazon Product Reviews (Ni, Li & McAuley, 2019), UCSD.
Source: https://nijianmo.github.io/amazon/index.html
Use the **ratings-only** file for the Electronics category (columns: item, user, rating, timestamp; no header).
The raw data is **not** stored in this repo. Download it into `data/` (see `data/README.md`).

## Pipeline
Ingest (explicit schema) -> clean -> filter (>=5 ratings per user/item) -> index IDs -> 70/15/15 split -> baselines -> ALS grid search -> test evaluation -> top-10 recs.

## Setup
Requirements: Python 3.10+, Java 11/17/21.

```bash
git clone https://github.com/Hexecutionerr/ecommerce-recommender-spark.git
cd ecommerce-recommender-spark
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
# Optional: check the pipeline works using a SYNTHETIC file (results are meaningless)
python scripts/make_synthetic_smoke_test.py 60000 data/synthetic_smoke_test.csv
python src/als_recommender.py --input data/synthetic_smoke_test.csv --output /tmp/smoke_out

# Quick run on a 5% sample of the real data
python src/als_recommender.py --input data/Electronics.csv --sample-fraction 0.05

# Full run (needs more RAM)
python src/als_recommender.py --input data/Electronics.csv --driver-memory 8g
```
Adjust the file name to match what you downloaded.

## Outputs
- `results/metrics.json`: dataset stats and RMSE for global mean, item mean and ALS
- `results/sample_recommendations.csv`: top-10 recommendations for sample users

## Results
Fill in from your own `results/metrics.json`:

| Model | Test RMSE |
|---|---|
| Global mean | |
| Item mean | |
| ALS | |

Run configuration: sample fraction = ___, min ratings = ___.

## Limitations
- RMSE evaluates rating prediction, not ranking quality.
- Cold-start users and items are dropped at evaluation.
- Small hyperparameter grid; no cross-validation.

## Future Work
Ranking metrics (Precision@K, NDCG), hybrid models using review text, implicit-feedback ALS, cluster deployment, streaming updates.

## Contributions
- Hasnain Khan: preprocessing, ALS implementation and tuning, repository, README
- Ali: dataset research, baselines, results analysis, blog

## References
See `blog/blog.md`.
