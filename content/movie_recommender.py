"""
Movie Recommender System - Content-Based Approach
================================================

This script demonstrates a content-based movie recommendation system using the MovieLens dataset.
It showcases key concepts in recommender systems including:
- Feature engineering and preprocessing
- Content-based filtering
- User and item profile creation
- Similarity computation
- Recommendation generation
"""

# Standard library imports
import os
import sys
import warnings
from typing import List, Tuple, Dict, Optional

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Local imports - Import the ContentBasedRecommender class
import importlib.util
spec = importlib.util.spec_from_file_location("content_based_recommender", "content-based_recommender.py")
content_based_recommender = importlib.util.module_from_spec(spec)
spec.loader.exec_module(content_based_recommender)
ContentBasedRecommender = content_based_recommender.ContentBasedRecommender

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_movielens_data():
    """
    Load and preprocess the MovieLens 100k dataset
    
    Returns:
    --------
    tuple: (movies_df, ratings_df) - Preprocessed movie and rating dataframes
    """
    print("Loading MovieLens 100k dataset...")
    
    try:
        # Fetch the MovieLens dataset from OpenML
        # This dataset contains 100,000 ratings from 943 users on 1682 movies
        movies = fetch_openml(name='movielens-100k', as_frame=True)
        movies_df = movies.frame
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {movies_df.shape}")
        print(f"Columns: {list(movies_df.columns)}")
        
        return movies_df
        
    except Exception as e:
        print(f"Error loading MovieLens dataset: {e}")
        print("Falling back to synthetic data...")
        return create_synthetic_data()

def create_synthetic_data():
    """
    Create synthetic movie data for demonstration purposes
    This is used as a fallback when MovieLens dataset is unavailable
    
    Returns:
    --------
    tuple: (movies_df, ratings_df) - Synthetic movie and rating dataframes
    """
    print("Creating synthetic movie dataset...")
    
    # Generate synthetic movie data
    n_movies = 100
    n_users = 50
    
    # Create movie features with realistic distributions
    genres = ['Action', 'Drama', 'Comedy', 'Thriller', 'Romance', 'Sci-Fi', 'Horror', 'Documentary']
    directors = ['Spielberg', 'Nolan', 'Tarantino', 'Scorsese', 'Cameron', 'Kubrick', 'Hitchcock', 'Fincher']
    
    movies_df = pd.DataFrame({
        'movie_id': range(n_movies),
        'title': [f'Movie_{i:03d}' for i in range(n_movies)],
        'genre': np.random.choice(genres, n_movies, p=[0.15, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05]),
        'year': np.random.randint(1990, 2024, n_movies),
        'rating': np.random.uniform(1, 10, n_movies),
        'budget': np.random.uniform(1, 200, n_movies),
        'director': np.random.choice(directors, n_movies),
        'description': [f'Description for movie {i:03d}' for i in range(n_movies)]
    })
    
    # Create synthetic ratings with user preference patterns
    ratings_data = []
    for user_id in range(n_users):
        # Each user rates 5-20 movies
        n_ratings = np.random.randint(5, 20)
        rated_movies = np.random.choice(n_movies, n_ratings, replace=False)
        
        for movie_id in rated_movies:
            movie = movies_df.iloc[movie_id]
            base_rating = 5
            
            # Simulate user preferences based on movie features
            # Users have genre preferences
            if movie['genre'] in ['Action', 'Thriller']:
                base_rating += np.random.normal(1, 1)
            elif movie['genre'] in ['Drama', 'Romance']:
                base_rating += np.random.normal(-1, 1)
            
            # Year preference (slight preference for newer movies)
            year_factor = (movie['year'] - 1990) / (2024 - 1990)
            base_rating += year_factor * 1.5
            
            # Add noise for realism
            rating = max(1, min(10, base_rating + np.random.normal(0, 1)))
            ratings_data.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating
            })
    
    ratings_df = pd.DataFrame(ratings_data)
    
    print(f"Synthetic dataset created:")
    print(f"  Movies: {n_movies}")
    print(f"  Users: {n_users}")
    print(f"  Ratings: {len(ratings_df)}")
    
    return movies_df, ratings_df

def preprocess_movie_data(movies_df):
    """
    Preprocess movie data by extracting features and cleaning text
    
    Parameters:
    -----------
    movies_df : pandas.DataFrame
        Raw movie dataframe
    
    Returns:
    --------
    pandas.DataFrame: Preprocessed movie dataframe
    """
    print("Preprocessing movie data...")
    
    # Create a copy to avoid modifying original data
    df = movies_df.copy()
    
    # Extract year from title if it exists in parentheses
    # Example: "Toy Story (1995)" -> year = 1995
    df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
    
    # Clean title by removing year in parentheses
    df['title_clean'] = df['title'].str.replace(r'\(\d{4}\)', '').str.strip()
    
    # Fill missing values
    df['year'].fillna(df['year'].median(), inplace=True)
    df['title_clean'].fillna('Unknown', inplace=True)
    
    # Add derived features
    df['decade'] = (df['year'] // 10) * 10  # Group by decades
    
    print(f"Preprocessing complete. Shape: {df.shape}")
    return df

def evaluate_recommendations(recommender, test_ratings, movies_df, n_recommendations=10):
    """
    Evaluate recommendation quality using common metrics
    
    Parameters:
    -----------
    recommender : ContentBasedRecommender
        Trained recommender model
    test_ratings : pandas.DataFrame
        Test set ratings
    movies_df : pandas.DataFrame
        Movie dataframe
    n_recommendations : int
        Number of recommendations to generate
    
    Returns:
    --------
    dict: Evaluation metrics
    """
    print("Evaluating recommendation quality...")
    
    predictions = []
    actuals = []
    
    # Sample users for evaluation
    test_users = test_ratings['user_id'].unique()[:20]  # Evaluate first 20 users
    
    for user_id in test_users:
        if user_id in recommender.user_profiles:
            # Get recommendations
            recommendations = recommender.recommend(user_id, n_recommendations)
            
            # Get actual ratings for these movies
            user_ratings = test_ratings[test_ratings['user_id'] == user_id]
            
            for item_idx, similarity in recommendations:
                movie_id = movies_df.iloc[item_idx]['movie_id']
                
                # Check if user actually rated this movie
                actual_rating = user_ratings[user_ratings['movie_id'] == movie_id]['rating']
                
                if not actual_rating.empty:
                    predictions.append(similarity * 5)  # Scale similarity to rating scale
                    actuals.append(actual_rating.iloc[0])
    
    if predictions:
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        
        print(f"Evaluation Results:")
        print(f"  MSE: {mse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  Number of predictions: {len(predictions)}")
        
        return {'mse': mse, 'mae': mae, 'n_predictions': len(predictions)}
    else:
        print("No predictions available for evaluation")
        return {}

def visualize_recommendations(recommender, movies_df, user_id=0, n_recommendations=10):
    """
    Visualize recommendations and user preferences
    
    Parameters:
    -----------
    recommender : ContentBasedRecommender
        Trained recommender model
    movies_df : pandas.DataFrame
        Movie dataframe
    user_id : int
        User ID to visualize
    n_recommendations : int
        Number of recommendations to show
    """
    print(f"Visualizing recommendations for User {user_id}...")
    
    # Get recommendations
    recommendations = recommender.recommend(user_id, n_recommendations)
    
    if not recommendations:
        print(f"No recommendations available for User {user_id}")
        return
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Recommendation similarities
    movie_titles = []
    similarities = []
    
    for item_idx, similarity in recommendations:
        movie = movies_df.iloc[item_idx]
        movie_titles.append(f"{movie['title'][:20]}...")
        similarities.append(similarity)
    
    ax1.barh(range(len(similarities)), similarities, color='skyblue')
    ax1.set_yticks(range(len(movie_titles)))
    ax1.set_yticklabels(movie_titles)
    ax1.set_xlabel('Similarity Score')
    ax1.set_title(f'Top {n_recommendations} Recommendations for User {user_id}')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Feature importance
    feature_importance = recommender.get_feature_importance(user_id, top_features=10)
    
    if feature_importance:
        features, importance = zip(*feature_importance)
        ax2.barh(range(len(importance)), importance, color='lightcoral')
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in features])
        ax2.set_xlabel('Feature Importance')
        ax2.set_title(f'Top Features for User {user_id}')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate the content-based movie recommender system
    """
    print("=" * 60)
    print("CONTENT-BASED MOVIE RECOMMENDER SYSTEM")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    print("\n1. DATA LOADING AND PREPROCESSING")
    print("-" * 40)
    
    try:
        # Try to load MovieLens dataset
        movies_df = load_movielens_data()
        # For MovieLens, we need to create ratings_df from the data
        # This is a simplified approach - in practice you'd load ratings separately
        ratings_df = create_synthetic_ratings(movies_df)
    except:
        # Fallback to synthetic data
        movies_df, ratings_df = create_synthetic_data()
    
    # Preprocess the movie data
    movies_df = preprocess_movie_data(movies_df)
    
    # Step 2: Initialize the recommender system
    print("\n2. INITIALIZING RECOMMENDER SYSTEM")
    print("-" * 40)
    
    # Create content-based recommender with cosine similarity
    movie_recommender = ContentBasedRecommender(similarity_metric='cosine')
    
    # Define feature columns for item profiles
    # These are the features that will be used to create movie profiles
    feature_columns = ['year', 'rating', 'budget']  # Numerical features
    text_columns = ['title_clean']  # Text features for TF-IDF
    
    print(f"Feature columns: {feature_columns}")
    print(f"Text columns: {text_columns}")
    
    # Step 3: Create item profiles
    print("\n3. CREATING ITEM PROFILES")
    print("-" * 40)
    
    # Create profiles for all movies based on their features
    item_profiles = movie_recommender.create_item_profiles(
        movies_df, 
        feature_columns, 
        text_columns
    )
    
    print(f"Item profiles created successfully!")
    print(f"Profile shape: {item_profiles.shape}")
    print(f"Number of features: {len(movie_recommender.feature_names)}")
    
    # Step 4: Create user profiles
    print("\n4. CREATING USER PROFILES")
    print("-" * 40)
    
    # Create profiles for all users based on their rating history
    user_profiles = movie_recommender.create_user_profiles(
        ratings_df, 
        movies_df,
        user_id_col='user_id',
        item_id_col='movie_id', 
        rating_col='rating'
    )
    
    print(f"User profiles created successfully!")
    print(f"Number of user profiles: {len(user_profiles)}")
    
    # Step 5: Generate recommendations
    print("\n5. GENERATING RECOMMENDATIONS")
    print("-" * 40)
    
    # Generate recommendations for a sample user
    test_user_id = 1
    n_recommendations = 10
    
    recommendations = movie_recommender.recommend(
        user_id=test_user_id, 
        n_recommendations=n_recommendations
    )
    
    print(f"Top {n_recommendations} recommendations for User {test_user_id}:")
    print("-" * 50)
    
    for i, (item_idx, similarity) in enumerate(recommendations):
        movie = movies_df.iloc[item_idx]
        print(f"{i+1:2d}. {movie['title']:<30} "
              f"({movie['genre']:<10}, {movie['year']:.0f}) "
              f"- Similarity: {similarity:.3f}")
    
    # Step 6: Analyze user preferences
    print("\n6. USER PREFERENCE ANALYSIS")
    print("-" * 40)
    
    # Get feature importance for the user
    feature_importance = movie_recommender.get_feature_importance(
        test_user_id, 
        top_features=10
    )
    
    print(f"Top 10 most important features for User {test_user_id}:")
    for feature, importance in feature_importance:
        print(f"  {feature:<25}: {importance:.3f}")
    
    # Step 7: Evaluate the system
    print("\n7. SYSTEM EVALUATION")
    print("-" * 40)
    
    # Split data for evaluation (simplified approach)
    train_ratings, test_ratings = train_test_split(
        ratings_df, 
        test_size=0.2, 
        random_state=42
    )
    
    # Recreate user profiles with training data only
    train_recommender = ContentBasedRecommender(similarity_metric='cosine')
    train_recommender.create_item_profiles(movies_df, feature_columns, text_columns)
    train_recommender.create_user_profiles(train_ratings, movies_df)
    
    # Evaluate recommendations
    evaluation_results = evaluate_recommendations(
        train_recommender, 
        test_ratings, 
        movies_df
    )
    
    # Step 8: Visualize results
    print("\n8. VISUALIZATION")
    print("-" * 40)
    
    # Visualize recommendations and user preferences
    visualize_recommendations(movie_recommender, movies_df, test_user_id)
    
    # Visualize user and item profiles in 2D space
    print("Visualizing user and item profiles...")
    movie_recommender.visualize_profiles(user_ids=[0, 1, 2], n_items=30)
    
    print("\n" + "=" * 60)
    print("RECOMMENDER SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 60)

def create_synthetic_ratings(movies_df):
    """
    Create synthetic ratings for MovieLens movies
    This is a simplified approach for demonstration
    """
    n_users = 50
    n_movies = len(movies_df)
    
    ratings_data = []
    for user_id in range(n_users):
        n_ratings = np.random.randint(5, 20)
        rated_movies = np.random.choice(n_movies, n_ratings, replace=False)
        
        for movie_idx in rated_movies:
            movie = movies_df.iloc[movie_idx]
            base_rating = 5
            
            # Simulate user preferences
            if movie['year'] > 2000:
                base_rating += np.random.normal(0.5, 1)
            
            rating = max(1, min(10, base_rating + np.random.normal(0, 1)))
            ratings_data.append({
                'user_id': user_id,
                'movie_id': movie_idx,
                'rating': rating
            })
    
    return pd.DataFrame(ratings_data)

if __name__ == "__main__":
    # Run the main demonstration
    main()