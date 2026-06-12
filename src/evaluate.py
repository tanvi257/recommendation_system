import numpy as np
import pandas as pd

def compute_rmse(actual, predicted):
    """
    Computes Root Mean Squared Error (RMSE).
    """
    return np.sqrt(np.mean((actual - predicted) ** 2))

def compute_mae(actual, predicted):
    """
    Computes Mean Absolute Error (MAE).
    """
    return np.mean(np.abs(actual - predicted))

def compute_ap_at_k(actual_ratings, predicted_ratings, k=10, rel_threshold=3.5):
    """
    Computes Average Precision @ K (AP@K) for a single user's test ratings.
    Relevance is defined as actual rating >= rel_threshold (default 3.5).
    """
    # Filter out users who have no relevant items in their test set
    total_relevant = sum(1 for r in actual_ratings if r >= rel_threshold)
    if total_relevant == 0:
        return None  # Exclude from MAP calculation as per standard practice
        
    # Zip actual and predicted ratings and sort by predicted rating in descending order
    paired = list(zip(predicted_ratings, actual_ratings))
    # Sort descending by predicted rating, tie-break by actual rating
    paired.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Take the top K recommendations
    top_k_paired = paired[:k]
    
    ap = 0.0
    rel_count = 0
    for idx, (pred, act) in enumerate(top_k_paired):
        is_rel = 1 if act >= rel_threshold else 0
        if is_rel:
            rel_count += 1
            precision_at_idx = rel_count / (idx + 1)
            ap += precision_at_idx
            
    # Normalize by the minimum of K and the total number of relevant items in the test set
    denominator = min(k, total_relevant)
    return ap / denominator if denominator > 0 else 0.0

def evaluate_predictions(test_df, predictions, rel_threshold=3.5, k=10, max_eval_users=250):
    """
    Computes RMSE, MAE, and MAP@10 across the test dataset.
    """
    actuals = test_df['rating'].values
    preds = np.clip(predictions, 1.0, 5.0)
    
    # 1. Compute rating prediction metrics
    rmse = compute_rmse(actuals, preds)
    mae = compute_mae(actuals, preds)
    
    # 2. Compute MAP@K
    # Group test set by user to evaluate ranking performance per user
    test_df_copy = test_df.copy()
    test_df_copy['pred_rating'] = preds
    
    # Sample users to evaluate MAP@10 to make computation extremely fast
    unique_users = test_df_copy['user_id'].unique()
    if max_eval_users and len(unique_users) > max_eval_users:
        np.random.seed(42)
        eval_users = np.random.choice(unique_users, max_eval_users, replace=False)
        test_df_copy = test_df_copy[test_df_copy['user_id'].isin(eval_users)]
        
    ap_scores = []
    
    for user_id, group in test_df_copy.groupby('user_id'):
        if len(group) < 2:
            # Skip users with insufficient test ratings to form a ranked list
            continue
        group_actuals = group['rating'].values
        group_preds = group['pred_rating'].values
        
        ap = compute_ap_at_k(group_actuals, group_preds, k=k, rel_threshold=rel_threshold)
        if ap is not None:
            ap_scores.append(ap)
            
    map_score = np.mean(ap_scores) if ap_scores else 0.0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAP@10': map_score,
        'evaluated_users_ranking': len(ap_scores)
    }

if __name__ == "__main__":
    # Test evaluation functions
    actuals = np.array([5.0, 4.0, 2.0, 3.5, 1.0])
    preds = np.array([4.5, 3.8, 1.5, 4.0, 2.0])
    test_df = pd.DataFrame({
        'user_id': [1, 1, 1, 1, 1],
        'movie_id': [101, 102, 103, 104, 105],
        'rating': actuals
    })
    
    results = evaluate_predictions(test_df, preds)
    print("Evaluation test results:")
    print(results)
