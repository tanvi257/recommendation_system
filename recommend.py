import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

def get_similar_movies_svd(model, movie_id, movie_df, n=5):
    """
    Finds the top N most similar movies to movie_id in the SVD latent space.
    """
    if model.model is None:
        raise ValueError("SVD Model has not been trained yet!")
        
    mapper = model.mapper
    if movie_id not in mapper.movie_to_idx:
        return pd.DataFrame() # Movie unseen during training
        
    m_idx = mapper.movie_to_idx[movie_id]
    
    # Extract item latent factors
    item_factors = model.model.item_factors.weight.detach().cpu().numpy()
    target_vector = item_factors[m_idx].reshape(1, -1)
    
    # Compute cosine similarity with all other movies
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(target_vector, item_factors).flatten()
    
    # Get top N+1 indices (excluding self)
    top_indices = np.argsort(similarities)[::-1]
    top_indices = [idx for idx in top_indices if idx != m_idx][:n]
    
    results = []
    for idx in top_indices:
        sim_movie_id = mapper.idx_to_movie[idx]
        title_info = movie_df[movie_df['movie_id'] == sim_movie_id]
        title = title_info['title'].values[0] if len(title_info) > 0 else "Unknown"
        year = title_info['year'].values[0] if len(title_info) > 0 else None
        
        results.append({
            'movie_id': sim_movie_id,
            'title': title,
            'year': year,
            'similarity': similarities[idx]
        })
        
    return pd.DataFrame(results)

def get_similar_movies_cf(model, movie_id, movie_df, n=5):
    """
    Finds the top N most similar movies to movie_id using the CF similarity matrix.
    """
    mapper = model.mapper
    if movie_id not in mapper.movie_to_idx:
        return pd.DataFrame()
        
    m_idx = mapper.movie_to_idx[movie_id]
    similarities = model.similarity_matrix[m_idx]
    
    top_indices = np.argsort(similarities)[::-1][:n]
    
    results = []
    for idx in top_indices:
        sim_movie_id = mapper.idx_to_movie[idx]
        title_info = movie_df[movie_df['movie_id'] == sim_movie_id]
        title = title_info['title'].values[0] if len(title_info) > 0 else "Unknown"
        year = title_info['year'].values[0] if len(title_info) > 0 else None
        
        results.append({
            'movie_id': sim_movie_id,
            'title': title,
            'year': year,
            'similarity': similarities[idx]
        })
        
    return pd.DataFrame(results)

def generate_top_k_recommendations(model, user_id, train_df, movie_df, k=10):
    """
    Generates the top K movie recommendations for a user.
    Predicts ratings for all movies the user has NOT rated in the training set,
    and returns the highest predicted ratings with explanations.
    """
    mapper = model.mapper
    if user_id not in mapper.user_to_idx:
        # User is cold-start (unseen), return popular movies as fallback
        print(f"User {user_id} is unseen. Returning popular movies as fallback.")
        popular_movies = train_df.groupby('movie_id').size().reset_index(name='count')
        popular_movies = popular_movies.sort_values(by='count', ascending=False).head(k)
        popular_movies = popular_movies.merge(movie_df, on='movie_id', how='left')
        popular_movies['predicted_rating'] = model.global_mean
        popular_movies['explanation'] = "Recommended because it is highly popular among other users (Cold Start Fallback)."
        return popular_movies[['movie_id', 'title', 'year', 'predicted_rating', 'explanation']]
        
    # Get movies already rated by user in training set
    user_train_ratings = train_df[train_df['user_id'] == user_id]
    rated_movie_ids = set(user_train_ratings['movie_id'].values)
    
    # Get all movies known to the model
    all_movie_ids = list(mapper.movie_to_idx.keys())
    
    # Determine unrated movies
    unrated_movie_ids = [m_id for m_id in all_movie_ids if m_id not in rated_movie_ids]
    
    if len(unrated_movie_ids) == 0:
        return pd.DataFrame()
        
    # Predict ratings for all unrated movies
    user_ids_repeating = [user_id] * len(unrated_movie_ids)
    predicted_ratings = model.predict(user_ids_repeating, unrated_movie_ids)
    
    # Rank them
    ranked_indices = np.argsort(predicted_ratings)[::-1][:k]
    
    rec_movies = [unrated_movie_ids[i] for i in ranked_indices]
    rec_ratings = [predicted_ratings[i] for i in ranked_indices]
    
    # Construct explanations
    explanations = []
    # Identify user's highly rated movies in training set (rating >= 4.0)
    user_high_ratings = user_train_ratings[user_train_ratings['rating'] >= 4.0].sort_values(by='rating', ascending=False)
    
    # We will use latent factor similarity or CF similarity to generate explanations
    for m_id in rec_movies:
        if len(user_high_ratings) == 0:
            explanations.append("Recommended based on your overall rating history.")
            continue
            
        # Find which of the user's high-rated movies is most similar to the recommended movie
        best_similar_movie = None
        best_similarity = -1.0
        
        for idx, row in user_high_ratings.iterrows():
            u_liked_id = int(row['movie_id'])
            # Compute similarity
            if hasattr(model, 'model') and model.model is not None:
                # SVD Latent similarity
                m1_idx = mapper.movie_to_idx[m_id]
                m2_idx = mapper.movie_to_idx[u_liked_id]
                item_factors = model.model.item_factors.weight.detach().cpu().numpy()
                v1, v2 = item_factors[m1_idx], item_factors[m2_idx]
                sim = 1.0 - cosine(v1, v2)
            else:
                # CF similarity
                m1_idx = mapper.movie_to_idx[m_id]
                m2_idx = mapper.movie_to_idx[u_liked_id]
                sim = model.similarity_matrix[m1_idx, m2_idx]
                
            if sim > best_similarity:
                best_similarity = sim
                best_similar_movie = u_liked_id
                
        if best_similar_movie is not None and best_similarity > 0.3:
            liked_title = movie_df[movie_df['movie_id'] == best_similar_movie]['title'].values[0]
            explanations.append(f"Similar to '{liked_title}' which you rated highly ({user_high_ratings[user_high_ratings['movie_id'] == best_similar_movie]['rating'].values[0]:.0f} stars).")
        else:
            explanations.append("Recommended because it matches your preferred genre/factors.")
            
    # Merge with movie titles
    results = []
    for i in range(len(rec_movies)):
        m_id = rec_movies[i]
        title_info = movie_df[movie_df['movie_id'] == m_id]
        title = title_info['title'].values[0] if len(title_info) > 0 else "Unknown"
        year = title_info['year'].values[0] if len(title_info) > 0 else None
        
        results.append({
            'movie_id': m_id,
            'title': title,
            'year': year,
            'predicted_rating': rec_ratings[i],
            'explanation': explanations[i]
        })
        
    return pd.DataFrame(results)

def analyze_success_failure_cases(model, test_df, train_df, movie_df, num_users=3):
    """
    Analyzes success and failure cases by selecting a few users in the test set,
    predicting their ratings, and identifying where our predictions aligned (success)
    or clashed (failure) with their actual ratings.
    """
    print("\n--- Success & Failure Cases Analysis ---")
    mapper = model.mapper
    
    # Filter test users who are also in the training mapping
    test_users = test_df[test_df['user_id'].isin(mapper.user_to_idx.keys())]['user_id'].unique()
    np.random.seed(42)
    selected_users = np.random.choice(test_users, num_users, replace=False)
    
    for user_id in selected_users:
        user_test = test_df[test_df['user_id'] == user_id]
        user_train = train_df[train_df['user_id'] == user_id]
        
        preds = model.predict([user_id] * len(user_test), user_test['movie_id'].values)
        user_test_copy = user_test.copy()
        user_test_copy['predicted'] = preds
        user_test_copy['error'] = np.abs(user_test_copy['rating'] - preds)
        
        # Merge with titles
        user_test_copy = user_test_copy.merge(movie_df, on='movie_id', how='left')
        user_train_movies = user_train.merge(movie_df, on='movie_id', how='left')
        
        print(f"\n=========================================")
        print(f"Analyzing User ID: {user_id}")
        print(f"Training History: Rated {len(user_train)} movies.")
        top_train = user_train_movies.sort_values(by='rating', ascending=False).head(3)
        print("  Top rated in train:")
        for idx, row in top_train.iterrows():
            print(f"    - '{row['title']}' ({row['rating']} stars)")
            
        # Success cases: high actual rating, high predicted rating (or error < 0.5)
        successes = user_test_copy[(user_test_copy['rating'] >= 4.0) & (user_test_copy['predicted'] >= 3.8)].sort_values(by='error')
        print("\n  Success Cases (Accurate Predictions of Liked Content):")
        if len(successes) > 0:
            for idx, row in successes.head(2).iterrows():
                print(f"    - '{row['title']}': Actual {row['rating']} stars, Predicted {row['predicted']:.2f} (Error: {row['error']:.2f})")
        else:
            print("    - None found matching criteria.")
            
        # Failure cases: low actual rating, high predicted rating (or error >= 1.5)
        failures = user_test_copy[(user_test_copy['rating'] <= 2.5) & (user_test_copy['predicted'] >= 3.5)].sort_values(by='error', ascending=False)
        print("\n  Failure Cases (False Positives - Recommended but Disliked):")
        if len(failures) > 0:
            for idx, row in failures.head(2).iterrows():
                print(f"    - '{row['title']}': Actual {row['rating']} stars, Predicted {row['predicted']:.2f} (Error: {row['error']:.2f})")
        else:
            print("    - None found matching criteria.")
            
        # Failure cases: high actual rating, low predicted rating
        false_negatives = user_test_copy[(user_test_copy['rating'] >= 4.0) & (user_test_copy['predicted'] <= 3.0)].sort_values(by='error', ascending=False)
        print("\n  False Negatives (Missed Opportunities - Liked but predicted low):")
        if len(false_negatives) > 0:
            for idx, row in false_negatives.head(2).iterrows():
                print(f"    - '{row['title']}': Actual {row['rating']} stars, Predicted {row['predicted']:.2f} (Error: {row['error']:.2f})")
        else:
            print("    - None found matching criteria.")
            
    print("=========================================")
