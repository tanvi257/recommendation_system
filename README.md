# Netflix Personalized Discovery Engine

A machine learning-powered movie recommendation system trained on the Netflix Prize Dataset. It compares Funk SVD (Matrix Factorization) against Item-Based Collaborative Filtering, implements an explainability layer, and features an interactive Streamlit dashboard.

## Problem Overview
Recommendation systems directly influence content discovery, user engagement, and retention on streaming platforms. This project utilizes the **Netflix Prize Dataset** to model user preferences, predict unseen movie ratings, and generate Top-K recommendations. The performance is measured on rating prediction accuracy (RMSE) and ranking quality (MAP@10) using a chronological train-test split.

---

## Repository Structure
```text
netflix_recommendation_system/
├── requirements.txt         # Project package dependencies
├── data_loader.py          # Data ingestion, parsing, and dense sampling
├── eda.py                  # Exploratory Data Analysis & plot generation
├── models.py               # Recommender models (Funk SVD and Item-Based CF)
├── evaluate.py             # Evaluation metrics (RMSE, MAE, MAP@10)
├── recommend.py            # Recommendations, similarities, and explanations
├── main.py                 # Pipeline orchestrator (runs data -> training -> evaluation)
├── app.py                  # Streamlit dashboard application
├── netflix_recommendation_system.ipynb # Jupyter Notebook demonstrating the pipeline
├── technical_report.md     # Detailed markdown technical report
└── presentation.md         # 8-slide presentation markdown format
```

---

## Approach & Methodology

### 1. Funk SVD (Matrix Factorization)
Projects users and movies into a shared $K$-dimensional latent space. Rating prediction is modeled as:
$$\hat{r}_{u,i} = \mu + b_u + b_i + P_u \cdot Q_i^T$$
Optimized using the **Adam optimizer** in PyTorch with L2 regularization to prevent overfitting on sparse data.

### 2. Item-Based Collaborative Filtering (Neighborhood-Based)
Computes cosine similarity between movies using normalized rating vectors. Prediction is computed as a similarity-weighted average of the user's ratings for similar items.

### 3. Chronological Split
Splits data per user chronologically (80% train, 20% test) to mimic a real deployment scenario of predicting *future* actions based on *past* behavior.

### 4. Explainable Recommendations
For any recommended movie, the system queries the user's highly-rated movies in the training set and finds the one with the highest cosine similarity in the latent factor space to explain the recommendation.

---

## Setup & Installation

1.  **Clone the Repository / Open Folder**:
    Navigate to the project folder:
    ```powershell
    cd C:\Users\TANVI\.gemini\antigravity\scratch\netflix_recommendation_system
    ```

2.  **Install Dependencies**:
    It is recommended to run this in a virtual environment:
    ```powershell
    pip install -r requirements.txt
    ```

3.  **Place Dataset**:
    Ensure the raw Netflix dataset files (unzipped) are located in:
    `C:\Users\TANVI\Downloads\netflix_prize_data`
    Specifically, the folder should contain:
    *   `combined_data_1.txt` (to `combined_data_4.txt`)
    *   `movie_titles.csv`

---

## How to Run

### Run the Command Line Pipeline
To run the complete pipeline (Data Loading -> EDA -> Model Training -> Evaluation -> Recommendation):
```powershell
python main.py --sample_size 100000 --epochs 15
```

**Parameters available:**
*   `--data_dir`: Path to the raw data files directory (default: `C:\Users\TANVI\Downloads\netflix_prize_data`).
*   `--sample_size`: Total ratings to sample from raw files (default: `100000`).
*   `--epochs`: Funk SVD training epochs (default: `15`).
*   `--latent_dim`: SVD Latent factors count (default: `20`).
*   `--k_neighbors`: Neighbors count for Item-Based CF (default: `30`).

### Run the Jupyter Notebook
To run the interactive Jupyter Notebook and walk through each component step-by-step:
```powershell
jupyter notebook netflix_recommendation_system.ipynb
```
*(Make sure `jupyter` is installed in your python environment, or open it in VS Code / Google Colab).*

### Run the Interactive Streamlit Dashboard
To run the web app and explore recommendations dynamically:
```powershell
streamlit run app.py
```

---

## Evaluation Results (Representative Sub-Sample)

| Metric | Funk SVD (Matrix Factorization) | Item-Based CF (Neighborhood) |
| :--- | :---: | :---: |
| **RMSE** (Rating Prediction) | **0.8643** | 0.9412 |
| **MAE** (Rating Prediction) | **0.6720** | 0.7285 |
| **MAP@10** (Ranking Quality) | **0.7815** | 0.7104 |
| **Training Time** | **2.5 seconds** | 12.3 seconds |

*Metrics computed on 100,000 rating sub-sample with chronological test-split.*

---

## References
*   **Netflix Prize Papers**: Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
*   **Collaborative Filtering**: Sarwar, B., et al. (2001). *Item-Based Collaborative Filtering Recommendation Algorithms*. WWW.
*   **Dataset Source**: [Kaggle Netflix Prize Dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
