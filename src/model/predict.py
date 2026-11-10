# src/model/predict.py

def predict(user_id, candidate_recipe_ids, context=None, top_n=10):
    """
    Predict function agreed upon in Week 1.
    Returns a list of dictionaries sorted by expected consumed enrichment score.
    """
    # Mock implementation for Week 2 unblocking
    mock_results = []
    for i, recipe_id in enumerate(candidate_recipe_ids[:top_n]):
        mock_results.append({
            "recipe_id": recipe_id,
            "score": 10.0 - (i * 0.5),          # Expected consumed enrichment score
            "p_choose": 0.85,                   # P(choose i)
            "enrichment": 8.5,                  # u_j(i)
            "temptation": 6.0                   # v_j(i)
        })
        
    return sorted(mock_results, key=lambda x: x["score"], reverse=True)
