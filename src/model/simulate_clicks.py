# src/model/simulate_clicks.py

import pandas as pd
import numpy as np

def simulate_rounds(ratings_df, recipes_df, catalog_size=20000, sample_size=14):
    """
    Simulates rounds for training/evaluation:
    - Defines a platform catalog (random subset of recipes).
    - If a rated recipe is outside the catalog, it's recorded as 'outside_option'.
    - Fast sampling using numpy random generator to handle large datasets.
    """
    rng = np.random.default_rng(42)
    
    # Ensure chronological order
    if 'timestamp' in ratings_df.columns:
        ratings_df = ratings_df.sort_values(by=['user_id', 'timestamp'])
    
    all_recipe_ids = recipes_df['recipe_id'].unique() if 'recipe_id' in recipes_df.columns else recipes_df.index.unique()
    
    # 1. Define platform catalog (random subset of recipes)
    if len(all_recipe_ids) > catalog_size:
        platform_catalog = set(rng.choice(all_recipe_ids, size=catalog_size, replace=False))
    else:
        platform_catalog = set(all_recipe_ids)
        
    catalog_array = np.array(list(platform_catalog))
    rounds_data = []

    for user_id, group in ratings_df.groupby('user_id'):
        for _, row in group.iterrows():
            recipe_id = row['recipe_id']
            
            # 2. Check if the recipe is outside the platform catalog (Cheat Meal / Outside Option)
            if recipe_id not in platform_catalog:
                chosen_id = 'outside_option'
                # Show 15 random items entirely from the platform catalog
                if len(catalog_array) >= 15:
                    shown_set = rng.choice(catalog_array, size=15, replace=False).tolist()
                else:
                    shown_set = catalog_array.tolist()
            else:
                chosen_id = recipe_id
                # Fast sampling: draw more than needed from catalog, then drop the chosen one
                pool_size = min(sample_size + 5, len(catalog_array))
                sampled_pool = rng.choice(catalog_array, size=pool_size, replace=False)
                
                # Drop the chosen recipe if it's in the pool, then take exactly `sample_size` (14)
                filtered_pool = [r for r in sampled_pool if r != chosen_id]
                if len(filtered_pool) >= sample_size:
                    sampled_others = rng.choice(filtered_pool, size=sample_size, replace=False).tolist()
                else:
                    sampled_others = filtered_pool
                    
                shown_set = sampled_others + [chosen_id]
                rng.shuffle(shown_set)
            
            rounds_data.append({
                "user_id": user_id,
                "timestamp": row.get('timestamp', None),
                "shown_recipe_ids": shown_set,
                "chosen_recipe_id": chosen_id,
                "rating": row.get('rating', None)
            })
            
    return pd.DataFrame(rounds_data)
