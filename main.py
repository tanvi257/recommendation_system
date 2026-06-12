import os
import argparse
import json
import time
import pandas as pd
from data_loader import load_sampled_data, load_movie_titles, train_test_split_chronological
from eda import run_eda
from models import FunkSVD, ItemBasedCF
from evaluate import evaluate_predictions
from recommend import generate_top_k_recommendations, analyze_success_failure_cases

def main():
    parser = argparse.ArgumentParser(description="Netflix Prize Recommendation System Pipeline")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\TANVI\Downloads\netflix_prize_data",
                        help="Path to directory containing Netflix raw files")
    parser.add_argument("--sample_size", type=int, default=100000,
                        help="Target total ratings for training subset")
    parser.add_argument("--min_user_ratings", type=int, default=30,
                        help="Minimum ratings per user for subset density")
    parser.add_argument("--min_movie_ratings", type=int, default=50,
                        help="Minimum ratings per movie for subset density")
    parser.add_argument("--latent_dim", type=int, default=20,
                        help="Latent factors dimension for Funk SVD")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Epochs for Funk SVD training")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate for Funk SVD")
    parser.add_argument("--reg", type=float, default=0.02,
                        help="Regularization parameter for Funk SVD")
    parser.add_argument("--k_neighbors", type=int, default=30,
                        help="Neighbors count for Item-Based CF")
    parser.add_argument("--output_dir", type=str, default="assets",
                        help="Output folder for EDA plots and results")
                        
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("==================================================")
    print("STARTING NETFLIX RECOMMENDATION SYSTEM PIPELINE")
    print("==================================================")
    
    # 1. Load data
    t0 = time.time()
    try:
        movie_df = load_movie_titles(args.data_dir)
        ratings_df = load_sampled_data(
            data_dir=args.data_dir,
            num_files=1, # Default to file 1 for quick execution, but handles all
            max_ratings_per_file=5000000,
            min_user_ratings=args.min_user_ratings,
            min_movie_ratings=args.min_movie_ratings,
            target_total_ratings=args.sample_size
        )
    except Exception as e:
        print(f"Error loading raw data: {e}")
        return
        
    print(f"Data loading and preprocessing finished in {time.time() - t0:.2f}s.")
    
    # 2. Run EDA
    run_eda(ratings_df, movie_df, output_dir=args.output_dir)
    
    # 3. Train-Test Split
    train_df, test_df = train_test_split_chronological(ratings_df, test_ratio=0.2)
    
    # 4. Train Models
    # Model 1: Funk SVD
    t_svd_start = time.time()
    svd_model = FunkSVD(
        latent_dim=args.latent_dim,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
        device=None
    )
    svd_model.fit(train_df)
    t_svd_train = time.time() - t_svd_start
    print(f"Funk SVD trained in {t_svd_train:.2f}s.")
    
    # Model 2: Item-Based CF
    t_cf_start = time.time()
    cf_model = ItemBasedCF(k_neighbors=args.k_neighbors)
    cf_model.fit(train_df)
    t_cf_train = time.time() - t_cf_start
    print(f"Item-Based CF trained in {t_cf_train:.2f}s.")
    
    # 5. Evaluate Models
    print("\nEvaluating Funk SVD...")
    svd_preds = svd_model.predict(test_df['user_id'].values, test_df['movie_id'].values)
    svd_metrics = evaluate_predictions(test_df, svd_preds)
    
    print("\nEvaluating Item-Based Collaborative Filtering...")
    cf_preds = cf_model.predict(test_df['user_id'].values, test_df['movie_id'].values)
    cf_metrics = evaluate_predictions(test_df, cf_preds)
    
    # 6. Print Comparison Table
    comparison_data = {
        'Metric': ['RMSE', 'MAE', 'MAP@10', 'Training Time (s)'],
        'Funk SVD (Matrix Factorization)': [
            f"{svd_metrics['RMSE']:.4f}",
            f"{svd_metrics['MAE']:.4f}",
            f"{svd_metrics['MAP@10']:.4f}",
            f"{t_svd_train:.2f}s"
        ],
        'Item-Based CF (Neighborhood)': [
            f"{cf_metrics['RMSE']:.4f}",
            f"{cf_metrics['MAE']:.4f}",
            f"{cf_metrics['MAP@10']:.4f}",
            f"{t_cf_train:.2f}s"
        ]
    }
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*50)
    print("              MODEL COMPARISON RESULTS            ")
    print("="*50)
    print(comparison_df.to_string(index=False))
    print("="*50)
    
    # Save metrics JSON
    metrics_summary = {
        'svd': {**svd_metrics, 'train_time_seconds': t_svd_train},
        'cf': {**cf_metrics, 'train_time_seconds': t_cf_train}
    }
    with open(os.path.join(args.output_dir, "model_comparison.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    # 7. Generate Sample Recommendations (for the first user in the test set)
    sample_user_id = test_df['user_id'].iloc[0]
    print(f"\nGenerating Top-10 recommendations for User ID: {sample_user_id} using SVD...")
    
    recs = generate_top_k_recommendations(svd_model, sample_user_id, train_df, movie_df, k=10)
    print(recs.to_string(index=False))
    
    # 8. Success/Failure Case Analysis
    analyze_success_failure_cases(svd_model, test_df, train_df, movie_df, num_users=2)
    
    print("\nPipeline execution complete! Output results stored in:", args.output_dir)

if __name__ == "__main__":
    main()
