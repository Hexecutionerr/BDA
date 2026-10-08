# Building a Scalable E-Commerce Product Recommender with Apache Spark ALS

**Authors:** Hasnain Khan (Roll No. 23102A0075) and Ali (Roll No. 23102A0076)
Computer Engineering (CMPN-A), Vidyalankar Institute of Technology, Mumbai | Subject: Big Data Analytics
**GitHub:** https://github.com/Hexecutionerr/ecommerce-recommender-spark

## Summary

We built a collaborative-filtering product recommender on Amazon product review data using Apache Spark's ALS (Alternating Least Squares) algorithm. The pipeline ingests raw ratings, cleans and filters them, trains a matrix-factorization model in parallel, and compares it with simple baselines using RMSE. This post explains the problem, architecture, code and evaluation approach. All code is in the linked repository.

## Introduction

Open any online store and the first thing you see is "Recommended for you". These suggestions drive a large share of engagement and sales. Behind them sit millions of users, millions of products and billions of interactions. That is a Big Data problem.

In this assignment we implemented a small but realistic version of that pipeline. We use Apache Spark so the same code can run on a laptop today and on a cluster tomorrow without a rewrite.

## Problem Statement

**Given a history of user-product ratings, predict which products a user is likely to rate highly but has not interacted with yet, and return a ranked top-N list per user.**

Three properties make this hard:

1. **Scale:** Review datasets contain tens of millions of ratings.
2. **Sparsity:** A typical user rates a tiny fraction of the catalogue, so the user-item matrix is almost entirely empty.
3. **Cold start:** New users and new items have no history.

## Why Big Data Is Required

The 5 V's apply directly:

| V | In this project |
|---|---|
| **Volume** | Millions of ratings across millions of users and products, too large for naive single-threaded Python loops |
| **Velocity** | Real stores generate new ratings and clicks continuously, so models must be retrained regularly |
| **Variety** | Ratings, timestamps, reviews, metadata and images |
| **Veracity** | Duplicate and noisy ratings, plus fake or sparse feedback, need cleaning |
| **Value** | Better recommendations improve engagement and revenue |

A single-machine pandas approach would struggle with the full matrix. Spark partitions the data across cores or machines and parallelises both preprocessing and training.

## Dataset

We use the **Amazon Product Reviews dataset (Ni, Li and McAuley, 2019), maintained by Julian McAuley's group at UC San Diego**:

- Page: https://nijianmo.github.io/amazon/index.html
- We use the **ratings-only** file for one category (we chose *Electronics*). Each row has an item ID, user ID, rating (1-5) and a Unix timestamp.
- Download links are on the page above. The files are CSVs with no header row.

The dataset is public and intended for research use. The raw data is not committed to GitHub because of its size. The README explains how to download it.

## Tools and Technologies

| Tool | Purpose |
|---|---|
| Apache Spark / PySpark | Distributed data processing and ML |
| Spark MLlib ALS | Matrix-factorization recommender |
| Python | Pipeline scripting |
| Google Colab or local machine | Execution environment |
| GitHub | Version control and code sharing |

## Architecture and Workflow

```
Raw ratings CSV
      |
      v
Spark ingestion (explicit schema)
      |
      v
Cleaning: drop nulls, deduplicate, validate rating range
      |
      v
Filtering: keep users and items with >= 5 ratings
      |
      v
Indexing: string IDs -> integer IDs
      |
      v
Split: train / validation / test (70/15/15)
      |
      +--> Baselines (global mean, item mean)
      v
ALS training + small hyperparameter search
      |
      v
Evaluation (RMSE) -> Top-10 recommendations -> saved outputs
```

## Data Preprocessing

1. **Explicit schema.** We define column types up front instead of letting Spark infer them, which avoids an extra full pass over the data.
2. **Cleaning.** We drop rows with nulls, keep ratings between 1 and 5, and remove duplicate user-item pairs.
3. **Density filter.** Users and items with fewer than 5 ratings carry too little signal and add noise, so we remove them.
4. **ID indexing.** Amazon IDs are strings, but ALS requires integer IDs. We use Spark's `StringIndexer` and keep the label mapping so recommendations can be translated back to real product IDs.
5. **Splitting.** A random 70/15/15 split with a fixed seed gives reproducible training, validation and test sets.

## Methodology: Collaborative Filtering with ALS

**Matrix factorization** approximates the sparse rating matrix **R** (users x items) as the product of two small dense matrices:

R ≈ U × Vᵀ

Each user and each item gets a latent vector of length *k* (the "rank"). A predicted rating is the dot product of a user vector and an item vector. The model learns these vectors by minimising squared error on known ratings plus a regularisation term (Koren et al., 2009).

**Why ALS?** Fixing one matrix makes the problem for the other a simple least-squares solve. ALS alternates between them. Each user's vector can be solved independently of the others, so the work parallelises naturally across a Spark cluster. This is why ALS is a standard choice for large-scale recommenders in Spark MLlib.

**Baselines.** To check that the model is actually learning something, we compare against:

- **Global mean:** always predict the average training rating.
- **Item mean:** predict each item's average training rating.

**Hyperparameters.** We search a small grid on the validation set: rank in {10, 20} and regParam in {0.05, 0.1}. The best configuration by validation RMSE is then evaluated once on the test set.

## Implementation

The main script is `src/als_recommender.py`. These are the key parts.

**Loading with an explicit schema:**

```python
schema = StructType([
    StructField("item", StringType()),
    StructField("user", StringType()),
    StructField("rating", FloatType()),
    StructField("timestamp", LongType()),
])
df = spark.read.csv(path, schema=schema, header=False)
```

**Cleaning and filtering:**

```python
df = (df.dropna().filter(F.col("rating").between(1, 5))
        .dropDuplicates(["user", "item"]))
users_ok = df.groupBy("user").count().filter("count >= 5").select("user")
items_ok = df.groupBy("item").count().filter("count >= 5").select("item")
df = df.join(users_ok, "user").join(items_ok, "item")
```

**Training ALS:**

```python
als = ALS(userCol="user_idx", itemCol="item_idx", ratingCol="rating",
          rank=rank, regParam=reg, maxIter=10,
          coldStartStrategy="drop", nonnegative=True, seed=42)
model = als.fit(train)
```

`coldStartStrategy="drop"` removes test rows for users or items unseen in training, so RMSE is not `NaN`. Note that this slightly changes which rows are evaluated.

**Top-N recommendations:**

```python
recs = model.recommendForAllUsers(10)
```

The full runnable code is in the repository.

## Results and Interpretation

The script writes `results/metrics.json` with dataset statistics and test RMSE for every method.

**Dataset statistics (after filtering):**

| Metric | Value |
|---|---|
| Ratings | `<from metrics.json>` |
| Users | `<from metrics.json>` |
| Items | `<from metrics.json>` |
| Sparsity | `<from metrics.json>` |

**Test RMSE (lower is better):**

| Model | RMSE |
|---|---|
| Global mean | `<from metrics.json>` |
| Item mean | `<from metrics.json>` |
| ALS (best: rank=`<r>`, regParam=`<lambda>`) | `<from metrics.json>` |

`<Write 3-5 sentences here based on what YOUR run showed: did ALS beat both baselines? By how much? If not, why (sparsity, small grid, sampling)?>`

`results/sample_recommendations.csv` lists top-10 product IDs for a few users. These are real model outputs, which can be inspected but not judged "correct" without further evaluation such as ranking metrics.

## Challenges

- **Memory.** Running the full dataset in local mode on a laptop can exhaust RAM. We added a `--sample-fraction` option and a configurable driver memory.
- **String IDs.** ALS needs integer IDs, so we added an indexing step with reverse mapping.
- **Cold start.** New users and items cannot be scored by ALS. We drop them at evaluation, and a production system would need a fallback such as popularity or content-based recommendations.
- **RMSE limitations.** RMSE measures rating prediction accuracy, not the quality of a top-N list. Ranking metrics such as Precision@K or NDCG better reflect user experience.
- **Implicit signals.** Ratings are explicit feedback. Real systems also use clicks and purchases, which need an implicit-feedback ALS variant (Hu et al., 2008).

## Future Scope

- Add ranking metrics (Precision@K, Recall@K, NDCG).
- Use `CrossValidator` for more rigorous tuning.
- Blend in review text or product metadata for content-based and hybrid recommendations.
- Add implicit-feedback ALS using purchase or click data.
- Run on a real cluster (Databricks, AWS EMR or Dataproc) to measure scaling.
- Use Spark Structured Streaming with Kafka for near-real-time updates.

## Conclusion

We built an end-to-end, reproducible Spark pipeline for e-commerce recommendation. It covers ingestion, cleaning, indexing, baselines, distributed ALS training, evaluation and top-N generation. The project shows why distributed processing matters for sparse, large-scale interaction data. The same code scales from a laptop sample to a cluster.

## Contribution Statement

| Author | Contributions |
|---|---|
| **Hasnain Khan** | Data ingestion and preprocessing, ALS model implementation and tuning, repository setup, README |
| **Ali** | Dataset research and selection, baseline implementation and evaluation, results analysis, blog writing and publishing |

Both authors jointly designed the architecture, reviewed the code and edited the final blog.

## Self-Declaration

**Self-Declaration: It took me 4 hour(s) and 30 minute(s) to solve the assignment, ensuring accuracy, completeness, and adherence to professional and ethical standards.**

## References

1. J. Ni, J. Li, J. McAuley, "Justifying recommendations using distantly-labeled reviews and fine-grained aspects," EMNLP-IJCNLP, 2019. Dataset page: https://nijianmo.github.io/amazon/index.html
2. Y. Koren, R. Bell, C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," IEEE Computer, vol. 42, no. 8, pp. 30-37, 2009.
3. Y. Hu, Y. Koren, C. Volinsky, "Collaborative Filtering for Implicit Feedback Datasets," IEEE ICDM, 2008.
4. M. Zaharia et al., "Apache Spark: A Unified Engine for Big Data Processing," Communications of the ACM, vol. 59, no. 11, 2016.
5. Apache Spark MLlib documentation, "Collaborative Filtering": https://spark.apache.org/docs/latest/ml-collaborative-filtering.html
