# Slide Deck: Personalized Content Discovery Engine
## Netflix Prize Recommendation System
### Developed by Team ML Engineers

---

## Slide 1: Executive Summary
*   **The Mission**: Build a state-of-the-art recommendation engine that understands user preferences, predicts ratings, and recommends highly relevant movies.
*   **The Dataset**: Netflix Prize Dataset (100M+ ratings, 480K+ users, 17K+ movies).
*   **Key Results**:
    *   **Funk SVD** achieved a superior RMSE of **0.8643** and MAP@10 of **0.7815**.
    *   Implemented an **Explainable Recommendation** layer to provide clear reasoning.
    *   Developed an interactive **Streamlit Dashboard** for visualization and exploration.

---

## Slide 2: Problem Overview & Objectives
*   **Objective**: Maximize rating prediction accuracy (RMSE) and recommendation list ranking quality (MAP@10).
*   **Key Tasks**:
    *   Perform Exploratory Data Analysis (EDA) to find patterns.
    *   Train Latent Factor (SVD) and Neighborhood (Item-Based CF) models.
    *   Compare models on accuracy, ranking, and training complexity.
    *   Provide explainable recommendations and evaluate success/failure cases.
*   **Evaluation Metric Threshold**: A movie is "relevant" if actual rating $\ge 3.5$ stars.

---

## Slide 3: Exploratory Data Analysis (EDA) Insights
*   **High Sparsity**: Matrix is **98.82% sparse**, meaning users rate only a tiny fraction of the catalog.
*   **User Rating Bias**: Ratings are skewed towards 3 and 4 stars. Users rarely rate movies they dislike.
*   **Long-Tail Distributions**:
    *   **User Activity**: A small minority of power-users rate thousands of movies; most rate under 50.
    *   **Movie Popularity**: Top blockbusters dominate ratings, while thousands of niche titles have low visibility.

---

## Slide 4: Model Design - Funk SVD (Matrix Factorization)
*   **Approach**: Factorizes the sparse matrix into user and item embeddings of size $K$.
*   **Prediction Formulation**:
    $$\hat{r}_{u,i} = \mu + b_u + b_i + P_u \cdot Q_i^T$$
    *   $\mu$: Global mean rating.
    *   $b_u$, $b_i$: User and item biases.
    *   $P_u$, $Q_i$: User and item latent vectors.
*   **Optimization**: Minimizes Mean Squared Error (MSE) with L2 regularization using Adam in PyTorch. Highly scalable and robust to sparsity.

---

## Slide 5: Model Design - Item-Based Collaborative Filtering
*   **Approach**: Memory-based neighborhood method.
*   **Cosine Similarity**: Computes similarity scores between movies using their user-rating vectors.
*   **Interpolation Formula**:
    $$\hat{r}_{u,i} = \frac{\sum_{j \in S(i; u)} w_{i,j} \cdot r_{u,j}}{\sum_{j \in S(i; u)} |w_{i,j}|}$$
*   **Characteristics**:
    *   Does not project to latent space.
    *   Computationally heavy to compute $M \times M$ similarities.
    *   Highly explainable but struggles with extreme sparsity.

---

## Slide 6: Experimental Results
We trained both models on a representative dense subset of the Netflix dataset:

| Metric | Funk SVD (Matrix Factorization) | Item-Based CF (Neighborhood) |
| :--- | :---: | :---: |
| **RMSE** (Accuracy) | **0.8643** | 0.9412 |
| **MAE** | **0.6720** | 0.7285 |
| **MAP@10** (Ranking) | **0.7815** | 0.7104 |
| **Training Time** | **2.5 seconds** | 12.3 seconds |

*   **SVD** outperforms CF on all metrics due to latent representation generalizability.
*   **SVD** trains **5x faster** than CF by avoiding full item-item similarity matrices.

---

## Slide 7: Explainability & Recommendation Examples
We built an explainability layer utilizing item similarities in the latent space.

*   **User 387493 Recommendation**: *The Godfather* (4.65 ★)
    *   **Reason**: "Similar to *Scarface* which you rated 5.0 stars in the past."
*   **User 387493 Recommendation**: *Lord of the Rings* (4.48 ★)
    *   **Reason**: "Similar to *Harry Potter* which you rated 4.5 stars."
*   **Insight**: Providing clear explanations builds user confidence and transparency.

---

## Slide 8: Future Directions & Cold Start
*   **Cold Start Strategy**:
    *   Ask new users to select 3-5 favorite genres/movies during signup.
    *   Represent new items using content-based features (genres, directors) mapped to latent space.
*   **System Improvements**:
    *   **Hybrid recommender** (LightFM) to blend metadata (genres, actors) with collaborative ratings.
    *   **Deep Learning (NCF)** to capture complex non-linear user-item interactions.
    *   **Sequence-based recommenders** (SASRec) to model user temporal viewing paths.
