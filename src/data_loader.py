import os
import pandas as pd
import numpy as np
from tqdm import tqdm

DEFAULT_DATA_DIR = r"C:\Users\TANVI\Downloads\netflix_prize_data"

def load_movie_titles(data_dir=DEFAULT_DATA_DIR):
    """
    Parses movie_titles.csv file line-by-line to correctly handle titles containing commas.
    Returns a pandas DataFrame.
    """
    file_path = os.path.join(data_dir, "movie_titles.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Movie titles file not found at {file_path}")
        
    print(f"Loading movie titles from {file_path}...")
    movie_id_list = []
    year_list = []
    title_list = []
    
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by comma at most twice: [MovieID, Year, Title]
            parts = line.split(',', 2)
            if len(parts) >= 3:
                try:
                    m_id = int(parts[0])
                    year_val = parts[1]
                    year = int(year_val) if year_val != 'NULL' and year_val.isdigit() else None
                    title = parts[2]
                    
                    movie_id_list.append(m_id)
                    year_list.append(year)
                    title_list.append(title)
                except ValueError:
                    # Skip header or malformed lines
                    continue
            elif len(parts) == 2:
                try:
                    m_id = int(parts[0])
                    year_val = parts[1]
                    year = int(year_val) if year_val != 'NULL' and year_val.isdigit() else None
                    
                    movie_id_list.append(m_id)
                    year_list.append(year)
                    title_list.append("")
                except ValueError:
                    continue
                    
    df = pd.DataFrame({
        'movie_id': movie_id_list,
        'year': year_list,
        'title': title_list
    })
    print(f"Loaded {len(df)} movie titles.")
    return df

def parse_ratings_file(file_path, max_ratings=None):
    """
    Parses a single combined_data_x.txt rating file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ratings file not found at {file_path}")
        
    print(f"Parsing ratings from {os.path.basename(file_path)}...")
    movie_ids = []
    user_ids = []
    ratings = []
    dates = []
    
    current_movie = None
    count = 0
    
    # Read file line by line
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(':'):
                current_movie = int(line[:-1])
            else:
                parts = line.split(',')
                user_ids.append(int(parts[0]))
                ratings.append(float(parts[1]))
                dates.append(parts[2])
                movie_ids.append(current_movie)
                count += 1
                if max_ratings and count >= max_ratings:
                    break
                    
    df = pd.DataFrame({
        'user_id': user_ids,
        'movie_id': movie_ids,
        'rating': ratings,
        'date': pd.to_datetime(dates)
    })
    return df

def load_sampled_data(data_dir=DEFAULT_DATA_DIR, num_files=1, max_ratings_per_file=5000000, 
                      min_user_ratings=50, min_movie_ratings=100, target_total_ratings=100000):
    """
    Loads ratings from multiple files and performs density-preserving sampling 
    to create a high-quality, dense subset for local model training.
    """
    dfs = []
    for i in range(1, num_files + 1):
        file_name = f"combined_data_{i}.txt"
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            df_part = parse_ratings_file(file_path, max_ratings=max_ratings_per_file)
            dfs.append(df_part)
        else:
            print(f"Warning: {file_name} not found. Skipping.")
            
    if not dfs:
        raise ValueError(f"No rating files found in directory: {data_dir}")
        
    ratings_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(ratings_df)} raw ratings in total.")
    
    # Filter to make it a dense subset
    print("Filtering to create a dense subset...")
    print(f"Initial shape: {ratings_df.shape}")
    
    # Iteratively filter users and movies to make sure we satisfy min ratings for both
    converged = False
    loop_count = 0
    filtered_df = ratings_df.copy()
    
    while not converged and loop_count < 10:
        prev_shape = filtered_df.shape
        
        # Filter users
        user_counts = filtered_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_user_ratings].index
        filtered_df = filtered_df[filtered_df['user_id'].isin(active_users)]
        
        # Filter movies
        movie_counts = filtered_df['movie_id'].value_counts()
        popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
        filtered_df = filtered_df[filtered_df['movie_id'].isin(popular_movies)]
        
        if filtered_df.shape == prev_shape:
            converged = True
        loop_count += 1
        
    print(f"Shape after density filters: {filtered_df.shape}")
    
    # If the dataset is still too large, sample it to the target size while preserving density
    if len(filtered_df) > target_total_ratings:
        print(f"Subsampling dense dataset to target size of {target_total_ratings}...")
        # We sample users, and keep all their ratings to preserve user interaction profiles
        user_counts = filtered_df['user_id'].value_counts()
        unique_users = user_counts.index.values
        np.random.seed(42)
        
        # Sample users until we accumulate enough ratings
        sampled_users = set()
        accumulated_ratings = 0
        shuffled_users = np.random.permutation(unique_users)
        
        for u in shuffled_users:
            sampled_users.add(u)
            accumulated_ratings += user_counts[u]
            if accumulated_ratings >= target_total_ratings:
                break
                
        filtered_df = filtered_df[filtered_df['user_id'].isin(sampled_users)]
        print(f"Sampled shape: {filtered_df.shape}")
        
    return filtered_df

def train_test_split_chronological(df, test_ratio=0.2):
    """
    Performs a chronological train-test split per user.
    This mimics real-world recommendation evaluation (predicting future ratings based on past actions).
    """
    print("Performing chronological train-test split...")
    df = df.sort_values(by=['user_id', 'date'])
    
    train_indices = []
    test_indices = []
    
    # Group by user and split chronologically
    for user_id, group in df.groupby('user_id'):
        n_ratings = len(group)
        n_test = max(1, int(n_ratings * test_ratio))
        n_train = n_ratings - n_test
        
        indices = group.index.tolist()
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])
        
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df

if __name__ == "__main__":
    # Test loader
    try:
        titles = load_movie_titles()
        data = load_sampled_data(target_total_ratings=50000)
        train, test = train_test_split_chronological(data)
        print("Data loaded successfully!")
        print(f"Unique Users: {data['user_id'].nunique()}, Unique Movies: {data['movie_id'].nunique()}")
    except Exception as e:
        print(f"Error during test: {e}")
