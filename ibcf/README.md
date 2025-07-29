# Movie Recommender System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-orange.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-red.svg)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/yourusername/Movie-Recommender)

A sophisticated Python implementation of a movie recommender system based on the MovieLens dataset. This system leverages advanced machine learning techniques to provide personalized movie recommendations through two distinct approaches.

## Overview

The Movie Recommender System offers two complementary recommendation engines:

1. **System I**: Popularity-based recommendations using statistical analysis
2. **System II**: Item-Based Collaborative Filtering (IBCF) for personalized suggestions

## Key Features

- **Data Processing**: Efficient loading and preprocessing of MovieLens dataset
- **Popularity Analysis**: Advanced movie popularity scoring based on rating volume and quality
- **Collaborative Filtering**: Sophisticated item-based recommendation algorithm
- **Personalized Recommendations**: Tailored movie suggestions for individual users
- **Robust Fallback**: Intelligent fallback to popular movies when personalized data is insufficient
- **Performance Optimized**: Memory-efficient implementation for large-scale datasets

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Movie-Recommender.git
   cd Movie-Recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the system**
   ```bash
   python movie_recommender.py
   ```

## Usage Guide

### Basic Usage

Execute the main script to generate recommendations:

```bash
python movie_recommender.py
```

This will:
- Display the top 10 most popular movies
- Generate the top 100 movies for System II recommendations
- Process the complete MovieLens dataset

### Advanced Usage

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

The system utilizes the comprehensive MovieLens dataset:

| Metric | Value |
|--------|-------|
| **Total Ratings** | ~1,000,000 |
| **Movies** | 3,706 |
| **Users** | 6,040 |
| **Rating Scale** | 1-5 stars |

## Technical Implementation

### System I: Popularity-Based Recommendations

The popularity engine employs a dual-metric approach for optimal movie selection:

#### Rating Volume Analysis
- **Minimum Threshold**: 1,000 ratings per movie
- **Purpose**: Ensures statistical reliability and broad appeal
- **Benefit**: Eliminates bias from movies with limited ratings

#### Quality Assessment
- **Metric**: Average rating calculation
- **Method**: Mean of all user ratings per movie
- **Ranking**: Movies sorted by average rating (highest first)

This hybrid approach balances **quantity** (wide interest) with **quality** (viewer satisfaction), ensuring recommendations are both well-regarded and broadly recognized.

### System II: Item-Based Collaborative Filtering (IBCF)

The IBCF system implements a sophisticated recommendation algorithm with multiple optimization layers:

#### Data Preprocessing
- **Centering**: Rating matrix normalization by row mean subtraction
- **Missing Values**: Intelligent handling of unrated movies
- **Normalization**: Enhanced comparison through rating standardization

#### Similarity Computation
- **Algorithm**: Transformed cosine similarity between movies
- **Filtering**: Minimum 3 common users for similarity calculation
- **Transformation**: Similarity scores normalized to [0,1] range
- **Optimization**: Top 30 similarities retained per movie

#### Prediction Engine
For each unrated movie *i*, the predicted rating follows:

$$prediction_i = \frac{\sum(S_{ij} \times w_j)}{\sum(S_{ij})}$$

Where:
- $S_{ij}$ = similarity between movies *i* and *j*
- $w_j$ = user's rating for movie *j*
- Summation over all rated movies *j*

#### Intelligent Fallback
- **Trigger**: < 10 personalized predictions available
- **Action**: Seamless transition to System I recommendations
- **Filtering**: Excludes user's previously rated movies
- **Guarantee**: Always provides 10 complete recommendations

## Performance Optimizations

- **Precomputed Similarity Matrix**: Cached for rapid recommendation generation
- **Top 100 Cache**: Popular movies pre-ranked for instant fallback
- **Memory Efficiency**: Optimized for large-scale dataset processing
- **Scalable Architecture**: Designed for production deployment

## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

<div align="center">

**Made for the data science community**

[![GitHub stars](https://img.shields.io/github/stars/darinz/Movie-Recommender?style=social)](https://github.com/yourusername/Movie-Recommender)
[![GitHub forks](https://img.shields.io/github/forks/darinz/Movie-Recommender?style=social)](https://github.com/yourusername/Movie-Recommender)

</div> 