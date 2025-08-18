# Recommender Systems Applications

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-green.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-purple.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/darinz/Recommender-Systems)

This repository contains a comprehensive collection of recommender systems implementations, algorithms, and applications. All code examples, interactive applications, and research implementations are organized in this `app/` folder for easy access and learning. Explore practical implementations of various recommendation algorithms, from basic collaborative filtering to advanced deep learning approaches.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- R (for R implementations)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/darinz/Recommender-Systems.git
   cd Recommender-Systems/app
   ```

2. **Install dependencies for specific systems**
   ```bash
   # For the interactive movie recommender app
   pip install streamlit pandas numpy
   
   # For the bandits system
   pip install numpy pandas matplotlib seaborn scikit-learn
   
   # For the content-based system
   cd content
   pip install -r requirements.txt
   
   # For the IBCF system
   cd ../ibcf
   pip install -r requirements.txt
   ```

3. **Run the interactive movie recommender app**
   ```bash
   cd movie-rec-app
   streamlit run movie_recommender_app.py
   ```

## Applications Overview

This `app/` folder contains the following recommender systems implementations:

### 1. Interactive Movie Recommender (`movie-rec-app/`)

A **Streamlit-based web application** that provides an interactive interface for movie recommendations using content-based filtering and collaborative filtering.

**Key Features:**
- **Interactive UI**: Rate movies and get instant recommendations
- **Visual Interface**: Movie posters and intuitive design with expandable sections
- **Real-time Recommendations**: Instant personalized suggestions
- **Responsive Design**: Works on desktop and mobile
- **External Data Integration**: Loads similarity matrices and movie metadata from external sources

**Technology Stack:**
- Frontend: Streamlit
- Data Processing: Pandas, NumPy
- Algorithm: Item-based collaborative filtering
- Data Sources: External similarity matrices and movie metadata
- Deployment: Streamlit Cloud

**Quick Start:**
```bash
cd movie-rec-app
streamlit run movie_recommender_app.py
```

### 2. Multi-Armed Bandit System (`bandits/`)

An advanced recommendation system using **multi-armed bandit algorithms** to balance exploration of new movies with exploitation of known good recommendations.

**Key Features:**
- **Bandit Algorithms**: Epsilon-greedy, UCB, Thompson sampling
- **Hybrid Approach**: Combines collaborative and content-based filtering
- **Cold-Start Handling**: Robust handling of new users and movies
- **Real-time Learning**: Continuously updates based on user feedback
- **Performance Evaluation**: Comprehensive metrics and visualization

**Algorithms Implemented:**
- **Epsilon-Greedy**: Balances exploration (ε) and exploitation (1-ε)
- **Upper Confidence Bound (UCB)**: Uses uncertainty estimates to guide exploration
- **Thompson Sampling**: Bayesian approach using posterior sampling
- **Hybrid Approach**: Combines multiple recommendation signals

**Usage Example:**
```python
from recommendation_system import MovieRecommendationBandit

# Initialize the system
recommender = MovieRecommendationBandit(
    num_movies=500,
    num_users=200,
    feature_dim=50,
    exploration_rate=0.1
)

# Get a recommendation using epsilon-greedy
movie_id = recommender.epsilon_greedy_recommend(user_id=1, epsilon=0.2)

# Receive user feedback
recommender.receive_rating(user_id=1, movie_id=movie_id, rating=4.5)
```

**Quick Start:**
```bash
cd bandits
python recommendation_system.py
```

### 3. Content-Based Recommender System (`content/`)

A **content-based movie recommendation system** that analyzes movie features (genre, year, director, etc.) to generate personalized recommendations. Features comprehensive educational documentation and robust implementation.

**Key Features:**
- Content-based filtering with multiple similarity metrics (cosine, euclidean, pearson)
- Advanced feature engineering (numerical, categorical, text features)
- User profile creation from rating history
- Educational focus with step-by-step explanations
- Visualization tools and evaluation metrics (MSE, MAE)
- Robust error handling and fallback mechanisms
- Both Python and R implementations

**Usage Example:**
```python
from content_based_recommender import ContentBasedRecommender

# Initialize and create profiles
recommender = ContentBasedRecommender(similarity_metric='cosine')
item_profiles = recommender.create_item_profiles(movies_df, ['year', 'rating'], ['title'])
user_profiles = recommender.create_user_profiles(ratings_df, movies_df)

# Generate recommendations
recommendations = recommender.recommend(user_id=1, n_recommendations=10)
```

**Quick Start:**
```bash
cd content
pip install -r requirements.txt
python movie_recommender.py
# or for R implementation
Rscript content-based_recommender.R
```

### 4. Collaborative Filtering (`cf/`)

A comprehensive implementation of **collaborative filtering algorithms** with both Python and R implementations, providing educational resources and practical examples.

**Key Features:**
- **Dual Implementation**: Both Python and R versions for educational purposes
- **Multiple Algorithms**: User-based and item-based collaborative filtering
- **Educational Focus**: Step-by-step explanations and documentation
- **Cross-language Comparison**: Compare implementations across programming languages
- **Practical Examples**: Real-world usage scenarios and best practices

**Usage Example:**
```python
from collaborative_filtering import CollaborativeFiltering

# Initialize the system
cf = CollaborativeFiltering()

# Load and prepare data
ratings_matrix = cf.load_data()

# Generate recommendations using user-based CF
recommendations = cf.user_based_recommendations(user_id=1, n_recommendations=10)

# Generate recommendations using item-based CF
item_recommendations = cf.item_based_recommendations(user_id=1, n_recommendations=10)
```

**Quick Start:**
```bash
cd cf
python collaborative_filtering.py
# or for R implementation
Rscript collaborative_filtering.R
```

### 5. Item-Based Collaborative Filtering (`ibcf/`)

A sophisticated implementation of **item-based collaborative filtering** using the MovieLens dataset with advanced preprocessing and optimization.

**Key Features:**
- **Data Processing**: Efficient loading and preprocessing of MovieLens dataset
- **Popularity Analysis**: Advanced movie popularity scoring
- **Collaborative Filtering**: Sophisticated item-based recommendation algorithm
- **Personalized Recommendations**: Tailored movie suggestions
- **Robust Fallback**: Intelligent fallback to popular movies
- **Performance Optimized**: Memory-efficient implementation

**Technical Implementation:**
- **System I**: Popularity-based recommendations using statistical analysis
- **System II**: Item-Based Collaborative Filtering (IBCF) for personalized suggestions
- **Similarity Computation**: Transformed cosine similarity between movies
- **Prediction Engine**: Weighted average approach for rating prediction

**Usage Example:**
```python
from movie_recommender import load_data, get_top_popular_movies, myIBCF

# Load and prepare data
ratings, movies = load_data()

# Get top popular movies
top_10 = get_top_popular_movies(ratings, movies)

# Create a new user profile
newuser = pd.Series(index=S.index, data=np.nan)
newuser['m1613'] =5  # User rated movie "m1613" with 5 stars
newuser['m1755'] =4  # User rated movie "m1755" with 4 stars

# Generate personalized recommendations
recommendations = myIBCF(newuser, S, top100_ranking)
```

**Quick Start:**
```bash
cd ibcf
pip install -r requirements.txt
python movie_recommender.py
```

### 6. User-Based vs Item-Based CF Comparison (`ubcf-ibcf/`)

A comprehensive comparison system that implements and evaluates both **user-based collaborative filtering (UBCF)** and **item-based collaborative filtering (IBCF)** algorithms, providing insights into their performance characteristics and use cases.

**Key Features:**
- **Dual Implementation**: Both UBCF and IBCF algorithms in Python and R
- **Performance Comparison**: Side-by-side evaluation of both approaches
- **Educational Focus**: Detailed explanations of differences and trade-offs
- **Cross-language Support**: Python and R implementations for comparison
- **Comprehensive Analysis**: Metrics, visualizations, and practical insights

**Algorithm Comparison:**
- **UBCF**: Finds similar users and recommends items they liked
- **IBCF**: Finds similar items and recommends based on user's item preferences
- **Performance Metrics**: Accuracy, coverage, diversity, and computational efficiency
- **Scalability Analysis**: Memory and time complexity comparisons

**Usage Example:**
```python
from ubcf_vs_ibcf import compare_recommendation_systems

# Compare UBCF vs IBCF performance
results = compare_recommendation_systems(
    ratings_matrix=ratings_data,
    test_size=0.2,
    n_recommendations=10
)

# Analyze results
print(f"UBCF Accuracy: {results['ubcf']['accuracy']}")
print(f"IBCF Accuracy: {results['ibcf']['accuracy']}")
```

**Quick Start:**
```bash
cd ubcf-ibcf
python ubcf_vs_ibcf.py
# or for R implementation
Rscript ubcf_vs_ibcf.R
```

### 7. Latent Factor Models (`latent/`)

Advanced **latent factor models** implementation featuring SVD, NMF, and SVD++ algorithms in both Python and R. These models learn low-dimensional representations of users and items to capture underlying patterns in rating data.

**Key Features:**
- **Multiple Algorithms**: SVD, Non-negative Matrix Factorization (NMF), and SVD++
- **Dual Implementation**: Both Python and R versions for educational comparison
- **Custom Latent Factor Model**: Stochastic Gradient Descent optimization with configurable parameters
- **Enhanced SVD++**: Advanced algorithm with implicit feedback handling
- **Comprehensive Evaluation**: MAE, RMSE, coverage metrics, and extensive visualizations
- **Educational Focus**: Detailed factor analysis and model comparison

**Algorithms Implemented:**
- **SVD (Singular Value Decomposition)**: Matrix factorization using singular value decomposition
- **NMF (Non-negative Matrix Factorization)**: Factorization with non-negative constraints for interpretable factors
- **SVD++**: Enhanced SVD with implicit feedback and sophisticated update rules
- **Custom Latent Factor Model**: SGD-based implementation with user/item biases

**Usage Example:**
```python
from latent_factor_models import LatentFactorModel, SVDppModel

# Create and train a latent factor model
model = LatentFactorModel(n_factors=10, learning_rate=0.01)
model.fit(ratings_df)

# Make predictions
prediction = model.predict(user_id, item_id)

# Get recommendations
recommendations = model.recommend(user_id, n_recommendations=5)

# Find similar items
similar_items = model.get_similar_items(item_id, n_similar=5)
```

**Quick Start:**
```bash
cd latent
python latent_factor_models.py
# or for R implementation
Rscript latent_factor_models.R
```

### 8. Recommender System Challenges Analysis (`challenge/`)

A comprehensive analysis toolkit for studying fundamental challenges in recommender systems, including cold start problems, data sparsity, popularity bias, and scalability issues. Provides both Python and R implementations for educational and research purposes.

**Key Features:**
- **Challenge Analysis**: Cold start, sparsity, popularity bias, and scalability assessment
- **Synthetic Data Generation**: Built-in challenge patterns for controlled experiments
- **Statistical Analysis**: Comprehensive metrics and impact measurements
- **Visualization Tools**: 12 different plots showing various challenge aspects
- **Bias Mitigation**: Multiple debiasing techniques (inverse popularity, square root, log)
- **Dual Implementation**: Both Python and R versions for educational comparison

**Challenges Analyzed:**
- **Cold Start Problem**: New users and items with limited rating history
- **Data Sparsity**: Missing ratings and their impact on recommendation quality
- **Popularity Bias**: Inequality in item popularity using Gini coefficients
- **Scalability Issues**: Computational complexity and memory requirements

**Usage Example:**
```python
from challenge_analysis import RecommenderSystemChallenges

# Initialize the analyzer
challenge_analyzer = RecommenderSystemChallenges()

# Analyze cold start problems
cold_start_stats, user_counts, item_counts = challenge_analyzer.analyze_cold_start(ratings_df)

# Analyze data sparsity
sparsity_stats, rating_matrix, rating_dist = challenge_analyzer.analyze_sparsity(ratings_df)

# Analyze popularity bias
popularity_stats, item_popularity, user_activity = challenge_analyzer.analyze_popularity_bias(ratings_df)

# Test bias mitigation techniques
bias_stats, debiased_methods = challenge_analyzer.analyze_bias_mitigation(ratings_df)
```

**Quick Start:**
```bash
cd challenge
python challenge_analysis.py
# or for R implementation
Rscript challenge_analysis.R
```

### 9. Deep Recommender Systems (`deep-rec-sys/`)

Advanced **deep learning-based recommender systems** implementing state-of-the-art neural network architectures for collaborative filtering and rating prediction. Features both Python (PyTorch) and R (Keras) implementations with comprehensive evaluation and visualization capabilities.

**Key Features:**
- **Multiple Architectures**: Neural Collaborative Filtering (NCF), Wide & Deep, and Neural Matrix Factorization (NeuMF)
- **Dual Implementation**: Both Python and R versions for educational comparison
- **Synthetic Data Generation**: Non-linear interaction patterns for testing model capabilities
- **Comprehensive Evaluation**: MAE, RMSE, training curves, and performance analysis
- **Advanced Visualization**: Training progress, model comparisons, and prediction analysis
- **GPU Support**: PyTorch implementation supports GPU acceleration

**Algorithms Implemented:**
- **Neural Collaborative Filtering (NCF)**: Combines matrix factorization with multi-layer perceptrons
- **Wide & Deep**: Linear models for memorization + neural networks for generalization
- **Neural Matrix Factorization (NeuMF)**: Integrates GMF and MLP components for enhanced performance

**Quick Start:**
```bash
cd deep-rec-sys
python deep_rec_sys.py
# or for R implementation
Rscript deep_rec_sys.R
```

## Algorithm Comparison

| System | Algorithm | Strengths | Use Case |
|--------|-----------|-----------|----------|
| **[Interactive App](movie-rec-app/)** | Content-based + IBCF | Real-time, visual interface | User-facing applications |
| **[Bandits](bandits/)** | Multi-armed bandits | Exploration vs exploitation, cold-start | Research, adaptive systems |
| **[Content-Based](content/)** | Content-based filtering | Feature interpretable, cold-start handling | Educational, feature-rich domains |
| **[CF](cf/)** | Collaborative filtering | Educational, cross-language | Learning and comparison |
| **[IBCF](ibcf/)** | Item-based CF | Proven performance, interpretable | Production systems |
| **[UBCF vs IBCF](ubcf-ibcf/)** | UBCF + IBCF comparison | Algorithm comparison, educational | Research and learning |
| **[Latent Factor Models](latent/)** | SVD, NMF, SVD++ | Matrix factorization, latent representations | Advanced research, production systems |
| **[Challenge Analysis](challenge/)** | Cold start, sparsity, bias analysis | Challenge assessment, bias mitigation | Research, system evaluation |
| **[Deep Recommender Systems](deep-rec-sys/)** | NCF, Wide & Deep, NeuMF | Deep learning, non-linear patterns | Advanced research, neural approaches |

## Getting Started

### For Interactive Use
```bash
cd movie-rec-app
streamlit run movie_recommender_app.py
```

### For Educational/Content-Based Learning
```bash
cd content
pip install -r requirements.txt
python movie_recommender.py
```

### For Collaborative Filtering Learning
```bash
cd cf
python collaborative_filtering.py
# or for R implementation
Rscript collaborative_filtering.R
```

### For Research/Development
```bash
cd bandits
python recommendation_system.py
```

### For Production Systems
```bash
cd ibcf
python movie_recommender.py
```

### For Algorithm Comparison
```bash
cd ubcf-ibcf
python ubcf_vs_ibcf.py
# or for R implementation
Rscript ubcf_vs_ibcf.R
```

### For Latent Factor Models
```bash
cd latent
python latent_factor_models.py
# or for R implementation
Rscript latent_factor_models.R
```

### For Challenge Analysis
```bash
cd challenge
python challenge_analysis.py
# or for R implementation
Rscript challenge_analysis.R
```

### For Deep Recommender Systems
```bash
cd deep-rec-sys
python deep_rec_sys.py
# or for R implementation
Rscript deep_rec_sys.R
```

## Dataset Information

The systems utilize the comprehensive MovieLens dataset:

| Metric | Value |
|--------|-------|
| **Total Ratings** | ~1,000,000 |
| **Movies** | 3,706 |
| **Users** | 6,040 |
| **Rating Scale** | 1-5 stars |

## Performance Characteristics

- **Scalability**: Handles thousands of movies and users
- **Real-time**: Updates recommendations instantly
- **Memory Efficient**: Uses sparse representations
- **Robust**: Handles missing data gracefully
- **Cold-start**: Effective handling of new users and movies

## Contributing

We welcome contributions! Please feel free to submit pull requests or open issues for bugs and feature requests.

**Areas for improvement:**
- Integration with real movie datasets
- Advanced feature engineering
- A/B testing framework
- Production deployment considerations
- Additional bandit algorithms