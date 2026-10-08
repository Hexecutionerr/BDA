"""E-commerce recommender using Spark ALS on Amazon ratings-only data."""
import argparse
import json
import os

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (StructType, StructField, StringType,
                               FloatType, LongType)
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to ratings-only CSV")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--sample-fraction", type=float, default=1.0,
                   help="Fraction of raw rows to use (0-1), for low-RAM machines")
    p.add_argument("--min-ratings", type=int, default=5)
    p.add_argument("--driver-memory", default="4g")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    spark = (SparkSession.builder.appName("EcommerceALS")
             .config("spark.driver.memory", args.driver_memory)
             .config("spark.sql.shuffle.partitions", "64")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # 1. Ingest with explicit schema (ratings-only file: item,user,rating,timestamp)
    schema = StructType([
        StructField("item", StringType()),
        StructField("user", StringType()),
        StructField("rating", FloatType()),
        StructField("timestamp", LongType()),
    ])
    df = spark.read.csv(args.input, schema=schema, header=False)
    if args.sample_fraction < 1.0:
        df = df.sample(fraction=args.sample_fraction, seed=42)

    # 2. Clean
    df = (df.dropna()
            .filter(F.col("rating").between(1, 5))
            .dropDuplicates(["user", "item"]))

    # 3. Density filter
    users_ok = (df.groupBy("user").count()
                  .filter(F.col("count") >= args.min_ratings).select("user"))
    items_ok = (df.groupBy("item").count()
                  .filter(F.col("count") >= args.min_ratings).select("item"))
    df = df.join(users_ok, "user").join(items_ok, "item").cache()

    # 4. Dataset statistics
    n_ratings = df.count()
    n_users = df.select("user").distinct().count()
    n_items = df.select("item").distinct().count()
    sparsity = 1.0 - n_ratings / (n_users * n_items)
    print(f"ratings={n_ratings} users={n_users} items={n_items} sparsity={sparsity:.6f}")

    # 5. Index string IDs to integers
    u_model = StringIndexer(inputCol="user", outputCol="user_f").fit(df)
    i_model = StringIndexer(inputCol="item", outputCol="item_f").fit(df)
    df = i_model.transform(u_model.transform(df))
    df = (df.withColumn("user_idx", F.col("user_f").cast("int"))
            .withColumn("item_idx", F.col("item_f").cast("int"))
            .select("user", "item", "user_idx", "item_idx", "rating"))

    # 6. Split 70/15/15
    train, val, test = df.randomSplit([0.7, 0.15, 0.15], seed=42)
    train.cache()

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")

    # 7. Baselines
    global_mean = train.agg(F.avg("rating")).first()[0]
    rmse_global = evaluator.evaluate(
        test.withColumn("prediction", F.lit(float(global_mean))))

    item_means = train.groupBy("item_idx").agg(F.avg("rating").alias("prediction"))
    test_item = (test.join(item_means, "item_idx", "left")
                     .fillna({"prediction": float(global_mean)}))
    rmse_item = evaluator.evaluate(test_item)

    # 8. ALS with small grid, selected on validation set
    best = {"rmse": float("inf")}
    for rank in [10, 20]:
        for reg in [0.05, 0.1]:
            als = ALS(userCol="user_idx", itemCol="item_idx", ratingCol="rating",
                      rank=rank, regParam=reg, maxIter=10,
                      coldStartStrategy="drop", nonnegative=True, seed=42)
            model = als.fit(train)
            v_rmse = evaluator.evaluate(model.transform(val))
            print(f"rank={rank} reg={reg} val_rmse={v_rmse:.4f}")
            if v_rmse < best["rmse"]:
                best = {"rmse": v_rmse, "rank": rank, "reg": reg, "model": model}

    rmse_als = evaluator.evaluate(best["model"].transform(test))
    print(f"TEST RMSE: global={rmse_global:.4f} item_mean={rmse_item:.4f} als={rmse_als:.4f}")

    # 9. Top-10 recommendations for a sample of users, mapped back to real IDs
    user_map = spark.createDataFrame(list(enumerate(u_model.labels)), ["user_idx", "user"])
    item_map = spark.createDataFrame(list(enumerate(i_model.labels)), ["item_idx", "item"])
    recs = (best["model"].recommendForAllUsers(10).limit(20)
            .select("user_idx", F.explode("recommendations").alias("rec"))
            .select("user_idx", F.col("rec.item_idx").alias("item_idx"),
                    F.col("rec.rating").alias("predicted_rating"))
            .join(user_map, "user_idx").join(item_map, "item_idx")
            .select("user", "item", "predicted_rating"))
    recs.toPandas().to_csv(os.path.join(args.output, "sample_recommendations.csv"),
                           index=False)

    # 10. Save metrics
    metrics = {
        "ratings": n_ratings, "users": n_users, "items": n_items,
        "sparsity": sparsity, "sample_fraction": args.sample_fraction,
        "min_ratings": args.min_ratings,
        "best_rank": best["rank"], "best_regParam": best["reg"],
        "val_rmse_best": best["rmse"],
        "test_rmse_global_mean": rmse_global,
        "test_rmse_item_mean": rmse_item,
        "test_rmse_als": rmse_als,
    }
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    spark.stop()


if __name__ == "__main__":
    main()
