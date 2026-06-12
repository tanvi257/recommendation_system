import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set page configuration for premium layout
st.set_page_config(
    page_title="Netflix personalized Discovery Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium dark-mode styling with glassmorphism and custom cards
st.markdown("""
<style>
    /* Dark theme overrides and global font styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0d0f12;
        color: #f1f3f5;
    }
    
    /* Premium Title Header styling */
    .header-container {
        background: linear-gradient(135deg, #e50914 0%, #800008 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(229, 9, 20, 0.2);
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        color: #f8f9fa;
        opacity: 0.9;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(25, 29, 36, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px 0 rgba(0,0,0,0.25);
    }
    
    /* Metrics display */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e50914;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Movie recommendation cards */
    .movie-card {
        background: linear-gradient(145deg, #1f2530 0%, #151922 100%);
        border-left: 5px solid #e50914;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(229, 9, 20, 0.3);
    }
    .movie-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
    }
    .movie-meta {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-bottom: 6px;
    }
    .movie-explanation {
        font-size: 0.9rem;
        font-style: italic;
        color: #2ecc71;
    }
    .movie-score {
        float: right;
        font-weight: bold;
        background: #e50914;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Imports from project files
from data_loader import load_sampled_data, load_movie_titles, train_test_split_chronological
from models import FunkSVD, ItemBasedCF
from evaluate import evaluate_predictions
from recommend import generate_top_k_recommendations, get_similar_movies_svd

DATA_DIR = r"C:\Users\TANVI\Downloads\netflix_prize_data"

# Sidebar controls
st.sidebar.markdown("## 🎬 Discovery Config")
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Configuration")
sample_size = st.sidebar.slider("Training Subset Size", min_value=10000, max_value=200000, value=50000, step=10000,
                               help="Size of sampled rating matrix for training. Larger size takes longer to train.")
epochs = st.sidebar.slider("SVD Epochs", min_value=5, max_value=30, value=12, step=1)
latent_dim = st.sidebar.slider("SVD Latent Factors", min_value=5, max_value=50, value=15, step=5)

# Layer 1 Caching: Cache the slow data loading and splitting step
@st.cache_data(show_spinner=False)
def load_and_split_data(sample_size):
    movie_df = load_movie_titles(DATA_DIR)
    
    # Load sampled data (min ratings are set to keep density high)
    ratings_df = load_sampled_data(
        data_dir=DATA_DIR,
        min_user_ratings=30,
        min_movie_ratings=50,
        target_total_ratings=sample_size
    )
    
    # Train-test split
    train_df, test_df = train_test_split_chronological(ratings_df, test_ratio=0.2)
    return train_df, test_df, movie_df

# Layer 2 Caching: Cache the model training and evaluation step
@st.cache_resource(show_spinner=False)
def train_and_evaluate_models(_train_df, _test_df, sample_size, epochs, latent_dim):
    # Train Funk SVD
    svd = FunkSVD(latent_dim=latent_dim, lr=0.005, reg=0.02, epochs=epochs)
    svd.fit(_train_df)
    
    # Train Item-Based CF
    cf = ItemBasedCF(k_neighbors=25)
    cf.fit(_train_df)
    
    # Compute evaluation metrics
    svd_preds = svd.predict(_test_df['user_id'].values, _test_df['movie_id'].values)
    svd_eval = evaluate_predictions(_test_df, svd_preds)
    
    cf_preds = cf.predict(_test_df['user_id'].values, _test_df['movie_id'].values)
    cf_eval = evaluate_predictions(_test_df, cf_preds)
    
    return svd, cf, svd_eval, cf_eval

# Header block
st.markdown("""
<div class="header-container">
    <div class="header-title">🎬 Netflix Discovery Engine</div>
    <div class="header-subtitle">Machine Learning Powered Personalized Content Recommender & Analytical Dashboard</div>
</div>
""", unsafe_allow_html=True)

# Try loading and training
try:
    with st.spinner("🚀 Initializing Discovery Engine & Training Recommenders (Funk SVD & Item-Based CF)... Please wait..."):
        t0 = time.time()
        # Layer 1: Data Loading (cached based on sample_size)
        train_df, test_df, movie_df = load_and_split_data(sample_size)
        
        # Layer 2: Model Training (cached based on epochs/latent_dim/sample_size)
        svd, cf, svd_eval, cf_eval = train_and_evaluate_models(train_df, test_df, sample_size, epochs, latent_dim)
        t_duration = time.time() - t0
except Exception as e:
    st.error(f"❌ Error loading dataset or training models: {e}")
    st.info("💡 Please verify that the Kaggle dataset files are fully extracted in `C:\\Users\\TANVI\\Downloads\\netflix_prize_data`.")
    st.stop()

# Layout tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Overview & EDA", 
    "📈 Recommender Performance", 
    "🎯 Personalized Discovery", 
    "🔍 Similarity Explorer",
    "🧠 System Architecture"
])

# ================= TAB 1: DATASET OVERVIEW & EDA =================
with tab1:
    st.markdown("### Exploratory Data Analysis & Sparsity Insights")
    
    n_users = train_df['user_id'].nunique()
    n_movies = train_df['movie_id'].nunique()
    n_ratings = len(train_df) + len(test_df)
    possible_ratings = n_users * n_movies
    sparsity = (1 - (n_ratings / possible_ratings)) * 100
    
    # Grid of metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="glass-card"><span class="metric-label">Unique Users</span><br><span class="metric-value">{n_users:,}</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="glass-card"><span class="metric-label">Unique Movies</span><br><span class="metric-value">{n_movies:,}</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="glass-card"><span class="metric-label">Total Ratings</span><br><span class="metric-value">{n_ratings:,}</span></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="glass-card"><span class="metric-label">Matrix Sparsity</span><br><span class="metric-value">{sparsity:.2f}%</span></div>', unsafe_allow_html=True)
        
    st.markdown("#### Key Data Characteristics")
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Rating distribution
        st.markdown("**Distribution of User Ratings (1-5 Stars)**")
        fig, ax = plt.subplots(figsize=(8, 4))
        df_all = pd.concat([train_df, test_df])
        rating_counts = df_all['rating'].value_counts().sort_index()
        colors = sns.color_palette("rocket", len(rating_counts))
        bars = ax.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='black', alpha=0.85)
        ax.set_ylabel("Count")
        ax.set_xlabel("Stars")
        # Add labels
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + (len(df_all)*0.01), f"{yval/len(df_all)*100:.1f}%", ha='center', va='bottom', fontsize=9)
        fig.patch.set_facecolor('#0d0f12')
        ax.set_facecolor('#1a202c')
        ax.spines['bottom'].set_color('#ffffff')
        ax.spines['left'].set_color('#ffffff')
        ax.xaxis.label.set_color('#ffffff')
        ax.yaxis.label.set_color('#ffffff')
        ax.tick_params(colors='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    with col_right:
        # Popularity
        st.markdown("**Movie Rating Counts (Power Law Distribution)**")
        movie_counts = df_all.groupby('movie_id').size().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(movie_counts)), movie_counts.values, color='#e50914', linewidth=3)
        ax.fill_between(range(len(movie_counts)), movie_counts.values, color='#e50914', alpha=0.2)
        ax.set_yscale('log')
        ax.set_xlabel("Ranked Movies (Most to Least Popular)")
        ax.set_ylabel("Number of Ratings (Log Scale)")
        fig.patch.set_facecolor('#0d0f12')
        ax.set_facecolor('#1a202c')
        ax.spines['bottom'].set_color('#ffffff')
        ax.spines['left'].set_color('#ffffff')
        ax.xaxis.label.set_color('#ffffff')
        ax.yaxis.label.set_color('#ffffff')
        ax.tick_params(colors='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ================= TAB 2: RECOMMENDER PERFORMANCE =================
with tab2:
    st.markdown("### Offline Model Evaluation & Comparison")
    st.markdown("We evaluate and compare a Latent Factor Model (Funk SVD) against a Neighborhood Model (Item-Based CF).")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("#### Funk SVD (Matrix Factorization)")
        st.markdown("Reduces high-dimensional user-movie interaction matrix into lower-dimensional user and item latent matrices.")
        st.markdown(f"**RMSE**: `{svd_eval['RMSE']:.4f}` (Lower is better rating prediction accuracy)")
        st.markdown(f"**MAP@10**: `{svd_eval['MAP@10']:.4f}` (Higher is better recommendation ranking quality)")
        
    with col_m2:
        st.markdown("#### Item-Based Collaborative Filtering")
        st.markdown("Computes movie cosine similarities directly on user-rating vectors and interpolates ratings of similar items.")
        st.markdown(f"**RMSE**: `{cf_eval['RMSE']:.4f}` (Lower is better)")
        st.markdown(f"**MAP@10**: `{cf_eval['MAP@10']:.4f}` (Higher is better)")
        
    # Table comparison
    st.markdown("#### Comparative Summary Table")
    comparison_df = pd.DataFrame({
        "Model": ["Funk SVD (Latent Factor)", "Item-Based CF (Neighborhood)"],
        "RMSE (Prediction Quality)": [svd_eval['RMSE'], cf_eval['RMSE']],
        "MAP@10 (Ranking Quality)": [svd_eval['MAP@10'], cf_eval['MAP@10']],
        "Evaluated Test Cases": [len(test_df), len(test_df)]
    })
    st.table(comparison_df.style.format({
        "RMSE (Prediction Quality)": "{:.4f}",
        "MAP@10 (Ranking Quality)": "{:.4f}"
    }))
    
    st.info("💡 **Trade-off Observation**: Funk SVD typically scales better and achieves lower RMSE because it captures latent dimensions, whereas Item-Based CF is highly explainable but can struggle when sparsity is extreme.")

# ================= TAB 3: PERSONALIZED DISCOVERY =================
with tab3:
    st.markdown("### Generate Personalized Content Recommendations")
    
    # Select user ID from test set
    test_users = list(test_df['user_id'].unique()[:50])
    selected_user = st.selectbox("Select a User ID to inspect:", test_users)
    
    if selected_user:
        # Show User's training history
        user_history = train_df[train_df['user_id'] == selected_user].merge(movie_df, on='movie_id', how='left')
        
        st.markdown("#### User Rating Profile (Highest Rated Movies in Training Set)")
        col_hist1, col_hist2 = st.columns([2, 3])
        
        with col_hist1:
            st.write(user_history.sort_values(by='rating', ascending=False)[['rating', 'title', 'year']].head(5))
            
        with col_hist2:
            # Generate recommendations using SVD
            recs = generate_top_k_recommendations(svd, selected_user, train_df, movie_df, k=5)
            
            st.markdown("#### Personalized Top 5 Recommendations (Funk SVD)")
            if len(recs) > 0:
                for idx, row in recs.iterrows():
                    st.markdown(f"""
                    <div class="movie-card">
                        <span class="movie-score">Pred: {row['predicted_rating']:.2f}★</span>
                        <div class="movie-title">{row['title']}</div>
                        <div class="movie-meta">Release Year: {int(row['year']) if not pd.isna(row['year']) else 'N/A'}</div>
                        <div class="movie-explanation">🎯 {row['explanation']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No recommendations generated.")

# ================= TAB 4: SIMILARITY EXPLORER =================
with tab4:
    st.markdown("### Discover Content Similarities (Item Cosine Similarity)")
    st.markdown("Search for any movie to discover similar items computed in the learned SVD latent space.")
    
    # Search box for movies
    movie_titles = list(movie_df['title'].unique())
    search_title = st.selectbox("Select or Type a Movie Title:", movie_titles, index=0)
    
    if search_title:
        # Find movie id
        movie_row = movie_df[movie_df['title'] == search_title].iloc[0]
        m_id = int(movie_row['movie_id'])
        
        st.markdown(f"Selected Movie: **{search_title}** ({int(movie_row['year']) if not pd.isna(movie_row['year']) else 'N/A'})")
        
        # Get similar movies
        similar_movies = get_similar_movies_svd(svd, m_id, movie_df, n=6)
        
        if len(similar_movies) > 0:
            st.markdown("#### Top 6 Most Similar Movies in SVD Latent Space:")
            cols = st.columns(3)
            for idx, row in similar_movies.iterrows():
                col_idx = idx % 3
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="movie-card" style="border-left-color: #2ecc71;">
                        <span class="movie-score">{row['similarity']:.2f} Similarity</span>
                        <div class="movie-title">{row['title']}</div>
                        <div class="movie-meta">Release Year: {int(row['year']) if not pd.isna(row['year']) else 'N/A'}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ This movie was unseen during training and has no latent similarity profile.")

# ================= TAB 5: SYSTEM ARCHITECTURE =================
with tab5:
    st.markdown("### Recommender System Architecture Flow")
    st.markdown("""
    The recommendation engine processes raw data and yields predictions using the following pipeline:
    """)
    
    st.markdown("""
    ```mermaid
    graph TD
        A[Raw Netflix Dataset] --> B[data_loader.py: Density Filtering]
        B --> C[train_test_split_chronological]
        C --> D[Training Set]
        C --> E[Test Set]
        
        D --> F[Funk SVD model training]
        D --> G[Item-Based CF similarity training]
        
        F --> H[Rating Predictor]
        G --> H
        
        H --> I[evaluate.py: RMSE & MAP@10]
        E --> I
        
        F --> J[recommend.py: Top-K recommendations]
        J --> K[Explanation Generator]
        K --> L[Streamlit Interface]
    ```
    """)
    
    st.markdown("""
    #### Architectural Highlights:
    1. **Chronological Evaluation**: We split the data chronologically per user, simulating the real-world scenario of predicting *future* actions based on *past* preferences.
    2. **Sparse Filtering**: We implement a density convergence filter to iteratively clean active users and popular items, maintaining recommendation density.
    3. **Explainability Layer**: Explanations find the user's highly rated historical items that exhibit the highest cosine similarity to the recommended items in the learned embeddings.
    """)
