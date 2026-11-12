"""Ranking metrics for the EatWise evaluation (src/eval/).

Definitions follow the Shared Terminology doc:
- `recommended`: ranked list of recipe ids (best first).
- `relevant`: set of recipe ids the user liked in the test period
  (rating >= 4).
"""
import math


def recall_at_k(recommended, relevant, k=20):
    """Share of the user's relevant recipes that appear in the top-k."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):
    """NDCG@k with binary relevance (higher when hits are ranked earlier)."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)  # rank 0 -> 1/log2(2) = 1
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg


def evaluate_rankings(recommendations, relevant_by_user, k_ndcg=10, k_recall=20):
    """Average NDCG@k_ndcg and Recall@k_recall over users.

    recommendations: {user_id: [recipe_id, ...]} ranked lists (real or mock).
    relevant_by_user: {user_id: {recipe_id, ...}} from the test period.
    Users without relevant test items are skipped; users without a
    recommendation list count as an empty list.
    """
    ndcgs, recalls = [], []
    for user_id, relevant in relevant_by_user.items():
        if not relevant:
            continue
        ranked = recommendations.get(user_id, [])
        ndcgs.append(ndcg_at_k(ranked, relevant, k_ndcg))
        recalls.append(recall_at_k(ranked, relevant, k_recall))
    if not ndcgs:
        return {"ndcg": 0.0, "recall": 0.0, "n_users": 0}
    return {
        "ndcg": sum(ndcgs) / len(ndcgs),
        "recall": sum(recalls) / len(recalls),
        "n_users": len(ndcgs),
    }


if __name__ == "__main__":
    # Hand-checked examples.
    assert recall_at_k(["A", "B", "C"], {"A"}, 3) == 1.0
    assert ndcg_at_k(["A", "B", "C"], {"A"}, 10) == 1.0
    assert abs(ndcg_at_k(["B", "A", "C"], {"A"}, 10) - 1 / math.log2(3)) < 1e-9
    assert ndcg_at_k(["B", "C", "D"], {"A"}, 10) == 0.0
    assert recall_at_k(["A", "B", "C"], {"A", "D"}, 3) == 0.5
    assert recall_at_k(["A"], set(), 3) == 0.0
    print("metrics.py: all checks passed")