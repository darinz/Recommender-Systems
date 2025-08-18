# Recommender Systems

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-green.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-purple.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/darinz/Recommender-Systems)

This repository explores the application and research of recommender systems, focusing on practical implementations and insights. It features movie recommender systems built using the MovieLens dataset, along with key research findings, performance evaluations, and key learnings. Dive in for a hands-on exploration of recommendation algorithms, data analysis, and system optimization.

## Repository Structure

- **`app/`** - Practical implementations and applications
- **`reference/`** - Documentation and learning resources  
- **`research/`** - Research findings and experimental results

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
git clone https://github.com/darinz/Recommender-Systems.git
cd Recommender-Systems

# Install dependencies for specific apps
cd app/[specific-app]
pip install -r requirements.txt
```

### Run Interactive App
```bash
cd app/movie-rec-app
streamlit run movie_recommender_app.py
```

## Applications (`app/`)

| Application | Algorithm | Key Features |
|-------------|-----------|--------------|
| **[Interactive App](app/movie-rec-app/)** | Content-based + IBCF | Streamlit UI, real-time recommendations |
| **[Bandits](app/bandits/)** | Multi-armed bandits | Epsilon-greedy, UCB, Thompson sampling |
| **[Content-Based](app/content/)** | Content-based filtering | Multiple similarity metrics, feature engineering |
| **[Collaborative Filtering](app/cf/)** | UBCF/IBCF | Python & R implementations |
| **[IBCF](app/ibcf/)** | Item-based CF | MovieLens dataset, popularity analysis |
| **[UBCF vs IBCF](app/ubcf-ibcf/)** | Comparison study | Performance analysis, educational focus |
| **[Latent Factor Models](app/latent/)** | SVD, NMF, SVD++ | Matrix factorization, SGD optimization |
| **[Challenge Analysis](app/challenge/)** | Cold start, sparsity, bias | Statistical analysis, bias mitigation |
| **[Deep Recommender Systems](app/deep-rec-sys/)** | NCF, Wide & Deep, NeuMF | PyTorch & Keras, neural architectures |

## Key Features

- **Multiple Algorithms**: Content-based, collaborative filtering, matrix factorization, deep learning
- **Dual Implementation**: Python and R versions for educational comparison
- **Interactive Interface**: Streamlit-based web application
- **Research Focus**: Comprehensive analysis of recommender system challenges
- **Educational**: Step-by-step explanations and documentation

## Dataset

MovieLens dataset with ~1M ratings, 3,706 movies, and 6,040 users (1-5 star scale).

## Getting Started

### Interactive Use
```bash
cd app/movie-rec-app
streamlit run movie_recommender_app.py
```

### Content-Based Learning
```bash
cd app/content
pip install -r requirements.txt
python movie_recommender.py
```

### Collaborative Filtering
```bash
cd app/cf
python collaborative_filtering.py
# or R: Rscript collaborative_filtering.R
```

### Research & Development
```bash
cd app/bandits
python recommendation_system.py
```

### Production Systems
```bash
cd app/ibcf
python movie_recommender.py
```

### Algorithm Comparison
```bash
cd app/ubcf-ibcf
python ubcf_vs_ibcf.py
```

### Latent Factor Models
```bash
cd app/latent
python latent_factor_models.py
```

### Challenge Analysis
```bash
cd app/challenge
python challenge_analysis.py
```

### Deep Learning
```bash
cd app/deep-rec-sys
python deep_rec_sys.py
```

## Performance

- **Scalability**: Handles thousands of movies and users
- **Real-time**: Instant recommendation updates
- **Memory Efficient**: Sparse representations
- **Robust**: Graceful handling of missing data
- **Cold-start**: Effective new user/item handling

## Contributing

We welcome contributions! Areas for improvement:
- Real movie dataset integration
- Advanced feature engineering
- A/B testing framework
- Production deployment
- Additional algorithms

## References

- Multi-armed bandit algorithms
- Collaborative filtering techniques
- Content-based recommendation methods
- Cold-start problem solutions
- Exploration vs exploitation trade-offs