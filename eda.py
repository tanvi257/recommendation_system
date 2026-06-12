import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_sampled_data, load_movie_titles

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

def run_eda(df, movie_df, output_dir="assets"):
    os.makedirs(output_dir, exist_ok=True)
    print("Running Exploratory Data Analysis...")
    
    # 1. Sparsity Analysis
    n_users = df['user_id'].nunique()
    n_movies = df['movie_id'].nunique()
    n_ratings = len(df)
    possible_ratings = n_users * n_movies
    sparsity = (1 - (n_ratings / possible_ratings)) * 100
    
    print(f"Number of Users: {n_users}")
    print(f"Number of Movies: {n_movies}")
    print(f"Number of Ratings: {n_ratings}")
    print(f"Sparsity: {sparsity:.4f}%")
    
    # Write summary stats
    with open(os.path.join(output_dir, "summary_statistics.txt"), "w") as f:
        f.write("Netflix Prize Dataset Subset Summary Statistics\n")
        f.write("==============================================\n")
        f.write(f"Unique Users: {n_users}\n")
        f.write(f"Unique Movies: {n_movies}\n")
        f.write(f"Total Ratings: {n_ratings}\n")
        f.write(f"Matrix Sparsity: {sparsity:.4f}%\n")
        f.write(f"Average Ratings per User: {n_ratings / n_users:.2f}\n")
        f.write(f"Average Ratings per Movie: {n_ratings / n_movies:.2f}\n")
        
    # 2. Rating Distribution Plot
    plt.figure(figsize=(8, 5))
    rating_counts = df['rating'].value_counts().sort_index()
    colors = sns.color_palette("viridis", len(rating_counts))
    bars = plt.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='black', alpha=0.85)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (n_ratings * 0.005), 
                 f"{yval:,} ({yval/n_ratings*100:.1f}%)", ha='center', va='bottom', fontsize=10)
                 
    plt.title("Distribution of Ratings", pad=15)
    plt.xlabel("Rating (Stars)")
    plt.ylabel("Count")
    plt.xticks(range(1, 6))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rating_distribution.png"), dpi=150)
    plt.close()
    
    # 3. User Activity (Ratings per User)
    user_ratings = df.groupby('user_id').size()
    plt.figure(figsize=(10, 5))
    sns.histplot(user_ratings.values, bins=50, kde=True, color='#2c3e50', log_scale=True)
    plt.axvline(user_ratings.median(), color='#e74c3c', linestyle='--', linewidth=2, 
                label=f'Median: {user_ratings.median():.0f}')
    plt.axvline(user_ratings.mean(), color='#27ae60', linestyle='-', linewidth=2, 
                label=f'Mean: {user_ratings.mean():.1f}')
    plt.title("User Activity Log-Distribution (Number of Ratings per User)", pad=15)
    plt.xlabel("Number of Ratings (Log Scale)")
    plt.ylabel("Number of Users")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "user_activity.png"), dpi=150)
    plt.close()
    
    # 4. Movie Popularity (Ratings per Movie)
    movie_ratings = df.groupby('movie_id').size()
    plt.figure(figsize=(10, 5))
    sns.histplot(movie_ratings.values, bins=50, kde=True, color='#2980b9', log_scale=True)
    plt.axvline(movie_ratings.median(), color='#e74c3c', linestyle='--', linewidth=2, 
                label=f'Median: {movie_ratings.median():.0f}')
    plt.axvline(movie_ratings.mean(), color='#27ae60', linestyle='-', linewidth=2, 
                label=f'Mean: {movie_ratings.mean():.1f}')
    plt.title("Movie Popularity Log-Distribution (Number of Ratings per Movie)", pad=15)
    plt.xlabel("Number of Ratings (Log Scale)")
    plt.ylabel("Number of Movies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "movie_popularity.png"), dpi=150)
    plt.close()
    
    # 5. Average Rating per Movie vs Number of Ratings
    movie_stats = df.groupby('movie_id').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    
    # Merge with titles to find top movies
    movie_stats = movie_stats.merge(movie_df, on='movie_id', how='left')
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=movie_stats, x='num_ratings', y='avg_rating', alpha=0.5, color='#8e44ad')
    plt.title("Average Rating vs. Movie Popularity (Number of Ratings)", pad=15)
    plt.xlabel("Number of Ratings")
    plt.ylabel("Average Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rating_vs_popularity.png"), dpi=150)
    plt.close()
    
    # Print interesting observations
    print("\n--- Key Insights from EDA ---")
    print(f"Top 5 Most Rated Movies:")
    top_rated = movie_stats.sort_values(by='num_ratings', ascending=False).head(5)
    for idx, row in top_rated.iterrows():
        print(f"  - {row['title']} ({row['year']}): {row['num_ratings']} ratings, Avg Rating: {row['avg_rating']:.2f}")
        
    print(f"\nTop 5 Highest Rated Movies (with at least 50 ratings):")
    highest_rated = movie_stats[movie_stats['num_ratings'] >= 50].sort_values(by='avg_rating', ascending=False).head(5)
    for idx, row in highest_rated.iterrows():
        print(f"  - {row['title']} ({row['year']}): Avg Rating: {row['avg_rating']:.2f}, {row['num_ratings']} ratings")
        
    print(f"\nSparsity of sampled subset: {sparsity:.2f}%")
    print(f"Plots saved to directory: {output_dir}")

if __name__ == "__main__":
    df = load_sampled_data(target_total_ratings=100000)
    movie_df = load_movie_titles()
    run_eda(df, movie_df)
