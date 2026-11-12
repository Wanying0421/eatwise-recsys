"""Evaluation harness for EatWise (Week 2).

Run from the repository root:

    python -m src.eval.run_eval                         # fake data, baselines
    python -m src.eval.run_eval --ratings data/processed/ratings.csv
    python -m src.eval.run_eval --predictions preds.json   # mock/real model output

Ratings table columns: user_id, recipe_id, rating, timestamp.
Predictions file: JSON {"user_id": ["recipe_id", ...]} (ranked, best first).
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.eval.metrics import evaluate_rankings

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remaining 15% = test
LIKED_RATING = 4                   # relevant = test-period rating >= 4
TOP_N = 20                         # length of each recommendation list


def make_fake_data(n_users=50, n_recipes=200, n_ratings=3000, seed=42):
    """Fake ratings table so the harness can run before real data exists."""
    rng = np.random.default_rng(seed)
    recipe_ids = np.array([f"r_{i}" for i in range(n_recipes)])
    popularity = 1.0 / np.arange(1, n_recipes + 1)  # a few popular recipes
    popularity /= popularity.sum()
    rating_bias = np.linspace(1.0, -1.0, n_recipes)  # popular ones rated higher

    users = rng.integers(0, n_users, n_ratings)
    recipe_idx = rng.choice(n_recipes, size=n_ratings, p=popularity)
    ratings = np.clip(np.round(3.5 + rating_bias[recipe_idx] * 0.8
                               + rng.normal(0, 1, n_ratings)), 1, 5).astype(int)
    start = pd.Timestamp("2018-01-01")
    timestamps = start + pd.to_timedelta(rng.integers(0, 365 * 24 * 3600, n_ratings), unit="s")

    df = pd.DataFrame({
        "user_id": [f"u_{u}" for u in users],
        "recipe_id": recipe_ids[recipe_idx],
        "rating": ratings,
        "timestamp": timestamps,
    })
    return df.drop_duplicates(["user_id", "recipe_id"]).reset_index(drop=True)


def temporal_split(df):
    """Sort by timestamp; first 70% train, next 15% validation, last 15% test."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    i_train, i_val = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    train, val, test = df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]
    # No-leakage check: everything in train happens before everything in test.
    assert train["timestamp"].max() <= test["timestamp"].min(), "temporal leakage!"
    return train, val, test


def relevant_items(test_df):
    """{user_id: set of recipes rated >= 4 in the test period}."""
    liked = test_df[test_df["rating"] >= LIKED_RATING]
    return liked.groupby("user_id")["recipe_id"].apply(set).to_dict()


def seen_items(train_df):
    """{user_id: recipes the user already rated in the training period}."""
    return train_df.groupby("user_id")["recipe_id"].apply(set).to_dict()


def random_baseline(users, all_recipes, seen, top_n=TOP_N, seed=0):
    rng = np.random.default_rng(seed)
    recs = {}
    for user in users:
        pool = rng.choice(all_recipes, size=min(len(all_recipes), top_n * 3), replace=False)
        recs[user] = [r for r in pool if r not in seen.get(user, set())][:top_n]
    return recs


def popular_baseline(users, train_df, seen, top_n=TOP_N):
    ranking = train_df["recipe_id"].value_counts().index.tolist()  # most rated first
    recs = {}
    for user in users:
        already = seen.get(user, set())
        recs[user] = [r for r in ranking if r not in already][:top_n]
    return recs


def print_table(results):
    print(f"\n{'method':<12}{'NDCG@10':>10}{'Recall@20':>12}{'users':>8}")
    for name, res in results.items():
        print(f"{name:<12}{res['ndcg']:>10.4f}{res['recall']:>12.4f}{res['n_users']:>8}")
    print()


def main():
    parser = argparse.ArgumentParser(description="EatWise evaluation harness")
    parser.add_argument("--ratings", help="CSV with user_id, recipe_id, rating, timestamp")
    parser.add_argument("--predictions", help="JSON {user_id: [recipe_id, ...]}")
    args = parser.parse_args()

    if args.ratings:
        df = pd.read_csv(args.ratings, parse_dates=["timestamp"])
    else:
        print("No --ratings given: using FAKE data.")
        df = make_fake_data()

    train, val, test = temporal_split(df)
    print(f"Split: train={len(train)}  val={len(val)}  test={len(test)}  "
          f"(train ends {train['timestamp'].max()}, test starts {test['timestamp'].min()})")

    relevant = relevant_items(test)
    seen = seen_items(train)
    users = list(relevant)
    all_recipes = df["recipe_id"].unique()

    results = {
        "random": evaluate_rankings(random_baseline(users, all_recipes, seen), relevant),
        "popular": evaluate_rankings(popular_baseline(users, train, seen), relevant),
    }
    if args.predictions:
        with open(args.predictions) as f:
            results["model"] = evaluate_rankings(json.load(f), relevant)
    print_table(results)


if __name__ == "__main__":
    main()