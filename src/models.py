import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class IDMapper:
    """
    Maps raw User ID and Movie ID to contiguous integer indices starting from 0,
    and handles inverse mappings.
    """
    def __init__(self):
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        
    def fit(self, user_ids, movie_ids):
        # Fit user mapping
        unique_users = sorted(list(set(user_ids)))
        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for idx, uid in enumerate(unique_users)}
        
        # Fit movie mapping
        unique_movies = sorted(list(set(movie_ids)))
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.idx_to_movie = {idx: mid for idx, mid in enumerate(unique_movies)}
        
    def map_users(self, user_ids):
        return np.array([self.user_to_idx[uid] if uid in self.user_to_idx else -1 for uid in user_ids])
        
    def map_movies(self, movie_ids):
        return np.array([self.movie_to_idx[mid] if mid in self.movie_to_idx else -1 for mid in movie_ids])


class RatingsDataset(Dataset):
    """
    PyTorch Dataset for user-movie ratings.
    """
    def __init__(self, users, items, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class FunkSVDNet(nn.Module):
    """
    PyTorch implementation of Funk SVD (Matrix Factorization with User and Item Biases).
    """
    def __init__(self, num_users, num_items, latent_dim=20, global_mean=3.5):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, latent_dim)
        self.item_factors = nn.Embedding(num_items, latent_dim)
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)
        
        # Initialize embeddings with small normal values and biases to 0
        nn.init.normal_(self.user_factors.weight, std=0.1)
        nn.init.normal_(self.item_factors.weight, std=0.1)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)
        
        self.global_mean = nn.Parameter(torch.tensor(global_mean), requires_grad=False)
        
    def forward(self, user_idx, item_idx):
        user_f = self.user_factors(user_idx)
        item_f = self.item_factors(item_idx)
        user_b = self.user_biases(user_idx).squeeze(-1)
        item_b = self.item_biases(item_idx).squeeze(-1)
        
        interaction = torch.sum(user_f * item_f, dim=-1)
        pred = self.global_mean + user_b + item_b + interaction
        return pred


class FunkSVD:
    """
    Wrapper class for Funk SVD model handling training, prediction, and mapping.
    """
    def __init__(self, latent_dim=20, lr=0.005, reg=0.02, epochs=15, batch_size=256, device=None):
        self.latent_dim = latent_dim
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.mapper = IDMapper()
        self.model = None
        self.global_mean = 3.5
        
    def fit(self, train_df):
        print(f"Training Funk SVD model on {self.device}...")
        self.mapper.fit(train_df['user_id'], train_df['movie_id'])
        self.global_mean = train_df['rating'].mean()
        
        num_users = len(self.mapper.user_to_idx)
        num_items = len(self.mapper.movie_to_idx)
        
        # Map IDs to indices
        user_idxs = self.mapper.map_users(train_df['user_id'])
        item_idxs = self.mapper.map_movies(train_df['movie_id'])
        ratings = train_df['rating'].values
        
        dataset = RatingsDataset(user_idxs, item_idxs, ratings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = FunkSVDNet(num_users, num_items, self.latent_dim, self.global_mean).to(self.device)
        # Disable weight_decay in Adam optimizer to prevent embedding parameters from decaying to zero
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for u, i, r in dataloader:
                u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(u, i)
                mse_loss = criterion(pred, r)
                
                # Manual L2 regularization for active embeddings in batch
                user_f = self.model.user_factors(u)
                item_f = self.model.item_factors(i)
                user_b = self.model.user_biases(u)
                item_b = self.model.item_biases(i)
                
                l2_reg = torch.sum(user_f ** 2) + torch.sum(item_f ** 2) + torch.sum(user_b ** 2) + torch.sum(item_b ** 2)
                loss = mse_loss + (self.reg / len(r)) * l2_reg
                
                loss.backward()
                optimizer.step()
                
                total_loss += mse_loss.item() * len(r)
                
            epoch_loss = total_loss / len(dataset)
            print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f}")
            
    def predict(self, user_ids, movie_ids):
        """
        Predicts ratings for a list of user IDs and movie IDs.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet!")
            
        self.model.eval()
        user_idxs = self.mapper.map_users(user_ids)
        item_idxs = self.mapper.map_movies(movie_ids)
        
        preds = np.full(len(user_ids), self.global_mean)
        
        # Separate into known and unknown combinations
        known_mask = (user_idxs != -1) & (item_idxs != -1)
        if not np.any(known_mask):
            return preds
            
        known_users = torch.tensor(user_idxs[known_mask], dtype=torch.long).to(self.device)
        known_items = torch.tensor(item_idxs[known_mask], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            known_preds = self.model(known_users, known_items).cpu().numpy()
            # Clip ratings to [1.0, 5.0]
            known_preds = np.clip(known_preds, 1.0, 5.0)
            
        preds[known_mask] = known_preds
        return preds


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering Model using Movie Cosine Similarity.
    """
    def __init__(self, k_neighbors=30):
        self.k_neighbors = k_neighbors
        self.mapper = IDMapper()
        self.similarity_matrix = None
        self.train_matrix = None
        self.global_mean = 3.5
        self.user_means = None
        
    def fit(self, train_df):
        print("Training Item-Based Collaborative Filtering...")
        self.mapper.fit(train_df['user_id'], train_df['movie_id'])
        self.global_mean = train_df['rating'].mean()
        
        num_users = len(self.mapper.user_to_idx)
        num_movies = len(self.mapper.movie_to_idx)
        
        # Create user-movie rating matrix
        user_idxs = self.mapper.map_users(train_df['user_id'])
        movie_idxs = self.mapper.map_movies(train_df['movie_id'])
        ratings = train_df['rating'].values
        
        self.train_matrix = np.zeros((num_users, num_movies))
        self.train_matrix[user_idxs, movie_idxs] = ratings
        
        # Compute user average ratings to normalize
        # We replace 0 ratings with NaN temporarily for average calculation
        train_matrix_nan = np.where(self.train_matrix == 0, np.nan, self.train_matrix)
        self.user_means = np.nanmean(train_matrix_nan, axis=1)
        # Fallback for users with no ratings to global mean
        self.user_means = np.nan_to_num(self.user_means, nan=self.global_mean)
        
        # Create normalized rating vectors for cosine similarity (subtracting user means)
        normalized_matrix = np.zeros_like(self.train_matrix)
        for u in range(num_users):
            rated_mask = self.train_matrix[u] > 0
            normalized_matrix[u, rated_mask] = self.train_matrix[u, rated_mask] - self.user_means[u]
            
        # Transpose to get movie rating vectors: shape (num_movies, num_users)
        movie_vectors = normalized_matrix.T
        
        # Compute Cosine Similarity between movies
        print("Computing movie similarity matrix...")
        self.similarity_matrix = cosine_similarity(movie_vectors)
        # Fill diagonal with 0 to avoid similarity to self
        np.fill_diagonal(self.similarity_matrix, 0)
        print("Similarity matrix computed.")
        
    def predict(self, user_ids, movie_ids):
        """
        Predicts ratings for a list of user IDs and movie IDs.
        """
        preds = []
        user_idxs = self.mapper.map_users(user_ids)
        movie_idxs = self.mapper.map_movies(movie_ids)
        
        for idx in range(len(user_ids)):
            u_idx = user_idxs[idx]
            m_idx = movie_idxs[idx]
            
            # Cold start fallback if user or movie is unseen during training
            if u_idx == -1 or m_idx == -1:
                preds.append(self.global_mean)
                continue
                
            # Get movies rated by this user
            user_ratings = self.train_matrix[u_idx]
            rated_movie_idxs = np.where(user_ratings > 0)[0]
            
            if len(rated_movie_idxs) == 0:
                preds.append(self.user_means[u_idx])
                continue
                
            # Get similarity of target movie to all movies rated by the user
            similarities = self.similarity_matrix[m_idx, rated_movie_idxs]
            ratings = user_ratings[rated_movie_idxs]
            
            # Get Top K neighbors
            top_k_indices = np.argsort(similarities)[-self.k_neighbors:]
            top_similarities = similarities[top_k_indices]
            top_ratings = ratings[top_k_indices]
            
            # Filter to positive similarities to maintain relevance
            pos_sim_mask = top_similarities > 0
            if not np.any(pos_sim_mask):
                # Fallback to user's average rating if no positive similarity neighbors exist
                preds.append(self.user_means[u_idx])
                continue
                
            top_similarities = top_similarities[pos_sim_mask]
            top_ratings = top_ratings[pos_sim_mask]
            
            # Weighted average rating prediction
            predicted_rating = np.sum(top_similarities * top_ratings) / np.sum(top_similarities)
            preds.append(np.clip(predicted_rating, 1.0, 5.0))
            
        return np.array(preds)
