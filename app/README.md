# Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-green.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Live-brightgreen.svg)](https://6oujescmadydjnbgygtm9o.streamlit.app)

A sophisticated **Movie Recommendation System** built with Streamlit that provides personalized movie recommendations using content-based filtering algorithms. The application leverages similarity scoring and collaborative filtering techniques to suggest movies based on user preferences.

## Live Demo

**[Try the Movie Recommender App](https://6oujescmadydjnbgygtm9o.streamlit.app)**

## Overview

This recommendation system implements content-based filtering using movie metadata including genres, keywords, cast, and other features. The algorithm computes similarity scores between movies and generates personalized recommendations based on user ratings.

### Key Features

- **Interactive Rating System**: Rate movies on a 1-5 scale to build your preference profile
- **Content-Based Filtering**: Recommendations based on movie similarity and metadata
- **Visual Interface**: Movie posters and intuitive UI powered by Streamlit
- **Real-time Recommendations**: Instant personalized suggestions based on your ratings
- **Responsive Design**: Optimized layout for different screen sizes

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Algorithm**: Item-based collaborative filtering
- **Data Format**: Parquet, CSV
- **Deployment**: Streamlit Cloud

## How It Works

1. **Data Loading**: The system loads pre-computed similarity matrices and movie metadata
2. **User Interaction**: Users rate movies from a curated selection to establish preferences
3. **Recommendation Generation**: The algorithm uses item-based collaborative filtering to predict ratings for unrated movies
4. **Results Display**: Top recommendations are presented with movie posters and titles

### Algorithm Details

The recommendation engine uses a weighted average approach:
- Computes similarity scores between movies using metadata features
- Predicts user ratings for unrated movies using weighted similarity scores
- Ranks movies by predicted ratings to generate top recommendations
- Falls back to popular movies if insufficient data is available

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Streamlit
- Pandas
- NumPy

### Installation

1. Clone the repository:
```bash
git clone https://github.com/darinz/Recommender-Systems.git
cd Recommender-Systems
```

2. Install dependencies:
```bash
pip install streamlit pandas numpy
```

3. Run the application:
```bash
streamlit run movie_recommender.py
```

### Usage

1. Open the application in your browser
2. Rate movies from the provided selection (1-5 stars)
3. Click "Get Recommendations" to receive personalized suggestions
4. Explore recommended movies with their posters and details

## Data Sources

The application uses:
- **Similarity Matrix**: Pre-computed similarity scores for top 100 movies
- **Movie Metadata**: Title, genres, cast, and poster URLs
- **External Data**: Sourced from public movie databases

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

**Ready to discover your next favorite movie?** [Launch the app](https://6oujescmadydjnbgygtm9o.streamlit.app)
