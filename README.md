# Recommender Systems

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-green.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-purple.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/darinz/Recommender-Systems)

This repository explores the application and research of recommender systems, focusing on practical implementations and insights. It features movie recommender system built using the MovieLens dataset, along with key research findings, performance evaluations, and key learnings. Dive in for a hands-on exploration of recommendation algorithms, data analysis, and system optimization.

## Project Structure

```
Recommender-Systems/
├── app/                    # Interactive Streamlit application
│   ├── movie_recommender.py
│   ├── data/
│   └── README.md
├── bandits/               # Multi-armed bandit algorithms
│   ├── recommendation_system.py
│   └── README.md
├── ibcf/                  # Item-based collaborative filtering
│   ├── movie_recommender.py
│   ├── requirements.txt
│   └── README.md
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/darinz/Recommender-Systems.git
   cd Recommender-Systems
   ```

2. **Install dependencies**
   ```bash
   # For the interactive app
   pip install streamlit pandas numpy
   
   # For the bandits system
   pip install numpy pandas matplotlib seaborn scikit-learn
   
   # For the IBCF system
   cd ibcf
   pip install -r requirements.txt
   ```

3. **Run the interactive app**
   ```bash
   cd app
   streamlit run movie_recommender.py
   ```

## Systems Overview

### 1. Interactive Movie Recommender (`app/`)

A **Streamlit-based web application** that provides an interactive interface for movie recommendations using content-based filtering.

**Key Features:**
- **Interactive UI**: Rate movies and get instant recommendations
- **Visual Interface**: Movie posters and intuitive design
- **Real-time Recommendations**: Instant personalized suggestions
- **Responsive Design**: Works on desktop and mobile

**Technology Stack:**
- Frontend: Streamlit
- Data Processing: Pandas, NumPy
- Algorithm: Item-based collaborative filtering
- Deployment: Streamlit Cloud

**[Try the Live Demo](https://6oujescmadydjnbgygtm9o.streamlit.app)**

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

### 3. Item-Based Collaborative Filtering (`ibcf/`)

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
newuser['m1613'] = 5  # User rated movie "m1613" with 5 stars
newuser['m1755'] = 4  # User rated movie "m1755" with 4 stars

# Generate personalized recommendations
recommendations = myIBCF(newuser, S, top100_ranking)
```

## Dataset Information

The systems utilize the comprehensive MovieLens dataset:

| Metric | Value |
|--------|-------|
| **Total Ratings** | ~1,000,000 |
| **Movies** | 3,706 |
| **Users** | 6,040 |
| **Rating Scale** | 1-5 stars |

## Algorithm Comparison

| System | Algorithm | Strengths | Use Case |
|--------|-----------|-----------|----------|
| **Interactive App** | Content-based + IBCF | Real-time, visual interface | User-facing applications |
| **Bandits** | Multi-armed bandits | Exploration vs exploitation, cold-start | Research, adaptive systems |
| **IBCF** | Item-based CF | Proven performance, interpretable | Production systems |

## Getting Started

### For Interactive Use
```bash
cd app
streamlit run movie_recommender.py
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

## References

- Multi-armed bandit algorithms for recommendation systems
- Collaborative filtering techniques
- Content-based recommendation methods
- Cold-start problem solutions
- Exploration vs exploitation trade-offs

---

**Ready to discover your next favorite movie?** [Launch the interactive app](https://6oujescmadydjnbgygtm9o.streamlit.app) or explore the different algorithms in this collection! 