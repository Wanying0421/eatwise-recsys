# src/model/simulate_clicks.py

import pandas as pd
import numpy as np

def simulate_rounds(ratings_df, recipes_df, sample_size=14):
    """
    Sorts each user's ratings by timestamp. Each rated recipe = chosen item of a round.
    The shown set S_j(t) = chosen recipe + 14 sampled others (15 items total).
    """
    # Ensure chronological order
    if 'timestamp' in ratings_df.columns:
        ratings_df = ratings_df.sort_values(by=['user_id', 'timestamp'])
    
    rounds_data = []
    all_recipe_ids = recipes_df['recipe_id'].unique() if 'recipe_id' in recipes_df.columns else recipes_df.index.unique()

    for user_id, group in ratings_df.groupby('user_id'):
        for _, row in group.iterrows():
            chosen_recipe = row['recipe_id']
            
            # Sample 14 other items to form a 15-item shown set S_j(t)
            other_candidates = [r for r in all_recipe_ids if r != chosen_recipe]
            sampled_others = np.random.choice(other_candidates, size=min(sample_size, len(other_candidates)), replace=False).tolist()
            
            shown_set = sampled_others + [chosen_recipe]
            np.random.shuffle(shown_set) # Shuffle so it's random order
            
            rounds_data.append({
                "user_id": user_id,
                "timestamp": row.get('timestamp', None),
                "shown_recipe_ids": shown_set,
                "chosen_recipe_id": chosen_recipe,
                "rating": row.get('rating', None)
            })
            
    return pd.DataFrame(rounds_data)
