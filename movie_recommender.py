import pandas as pd
import numpy as np
from typing import List, Optional
import requests
from io import StringIO

def load_data():
    """
    Load and preprocess the MovieLens dataset.
    
    Returns:
        tuple: (ratings DataFrame, movies DataFrame)
        - ratings: Contains UserID, MovieID, Rating, and Timestamp
        - movies: Contains MovieID, Title, Genres, Year, and image_url
    """
    # Read ratings data from URL
    ratings_url = 'https://liangfgithub.github.io/MovieData/ratings.dat?raw=true'
    ratings = pd.read_csv(ratings_url, 
                         sep=':',  # Use colon as delimiter
                         header=None,  # No header row
                         names=['UserID', 'MovieID', 'Rating', 'Timestamp'])  # Assign column names
    
    # Read movies data from URL
    movies_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
    movies = pd.read_csv(movies_url, 
                        sep='::',  # Use double colon as delimiter
                        header=None,  # No header row
                        names=['MovieID', 'Title', 'Genres'],  # Assign column names
                        encoding='latin1')  # Handle special characters
    
    # Convert MovieID to integer for consistency
    movies['MovieID'] = movies['MovieID'].astype(int)
    
    # Extract year from title (last 4 characters before the closing parenthesis)
    movies['Year'] = movies['Title'].str.extract(r'\((\d{4})\)').astype(int)
    
    # Generate image URLs for each movie
    base_url = 'https://liangfgithub.github.io/MovieImages/'
    movies['image_url'] = movies['MovieID'].apply(
        lambda x: f'{base_url}{x}.jpg?raw=true'
    )
    
    return ratings, movies

def get_top_popular_movies(ratings, movies, min_ratings=1000, top_n=10):
    """
    Get the top N most popular movies based on number of ratings and average rating.
    
    Args:
        ratings (DataFrame): Movie ratings data
        movies (DataFrame): Movie metadata
        min_ratings (int): Minimum number of ratings required (default: 1000)
        top_n (int): Number of top movies to return (default: 10)
    
    Returns:
        DataFrame: Top N popular movies with their details
    """
    # Calculate movie statistics
    movie_stats = ratings.groupby('MovieID').agg(
        ratings_count=('Rating', 'count'),
        avg_rating=('Rating', 'mean')
    ).reset_index()
    
    # Filter movies with minimum ratings and join with movie details
    popular_movies = (movie_stats[movie_stats['ratings_count'] >= min_ratings]
                     .merge(movies, on='MovieID')
                     .sort_values('avg_rating', ascending=False)
                     .head(top_n))
    
    return popular_movies

def get_top_100_movies(ratings, movies, min_ratings=1000):
    """
    Get the top 100 most popular movies for System II fallback.
    
    Args:
        ratings (DataFrame): Movie ratings data
        movies (DataFrame): Movie metadata
        min_ratings (int): Minimum number of ratings required
    
    Returns:
        DataFrame: Top 100 popular movies with their details
    """
    # Calculate movie statistics
    movie_stats = ratings.groupby('MovieID').agg(
        ratings_count=('Rating', 'count'),
        avg_rating=('Rating', 'mean')
    ).reset_index()
    
    # Filter and get top 100 movies
    top_100 = (movie_stats[movie_stats['ratings_count'] >= min_ratings]
               .merge(movies, on='MovieID')
               .sort_values('avg_rating', ascending=False)
               .head(100))
    
    # Save to CSV for later use
    top_100.to_csv('top100_high_rated.csv', index=False)
    
    return top_100

def transformed_cosine_similarity(Ri, Rj):
    """
    Calculate the transformed cosine similarity between two movies.
    
    Args:
        Ri (Series): Ratings for movie i
        Rj (Series): Ratings for movie j
    
    Returns:
        float: Transformed cosine similarity between 0 and 1, or None if insufficient data
    """
    # Find common users who rated both movies
    common_users = (Ri.notna() & Rj.notna())
    
    # If fewer than 3 common users, return None
    if common_users.sum() < 3:
        return None
    
    # Get ratings for common users
    Ri_common = Ri[common_users]
    Rj_common = Rj[common_users]
    
    # Calculate dot product and magnitudes
    dot_product = (Ri_common * Rj_common).sum()
    magnitude_i = np.sqrt((Ri_common * Ri_common).sum())
    magnitude_j = np.sqrt((Rj_common * Rj_common).sum())
    
    # Return None if either magnitude is zero
    if magnitude_i == 0 or magnitude_j == 0:
        return None
    
    # Calculate cosine similarity and transform to [0,1] range
    cosine_sim = dot_product / (magnitude_i * magnitude_j)
    return 0.5 + 0.5 * cosine_sim

def compute_similarity_matrix(ratings_matrix):
    """
    Compute the similarity matrix for all movies.
    
    Args:
        ratings_matrix (DataFrame): User-movie rating matrix
    
    Returns:
        DataFrame: Similarity matrix between movies
    """
    # Initialize similarity matrix
    n_movies = ratings_matrix.shape[1]
    S = pd.DataFrame(index=ratings_matrix.columns, 
                    columns=ratings_matrix.columns,
                    dtype=float)
    
    # Calculate similarities for each pair of movies
    for i in range(n_movies):
        for j in range(i, n_movies):
            sim = transformed_cosine_similarity(
                ratings_matrix.iloc[:, i],
                ratings_matrix.iloc[:, j]
            )
            S.iloc[i, j] = sim
            S.iloc[j, i] = sim  # Matrix is symmetric
    
    # Set diagonal to NaN (movie with itself)
    np.fill_diagonal(S.values, np.nan)
    
    return S

def get_top_30_similarities(S):
    """
    Keep only the top 30 similarities for each movie.
    
    Args:
        S (DataFrame): Full similarity matrix
    
    Returns:
        DataFrame: Similarity matrix with only top 30 similarities per movie
    """
    # Initialize new matrix
    S_top30 = pd.DataFrame(index=S.index, columns=S.columns)
    
    # For each movie, keep only top 30 similarities
    for movie in S.index:
        top_30 = S.loc[movie].nlargest(30, keep='first')
        S_top30.loc[movie, top_30.index] = top_30
    
    return S_top30

def myIBCF(newuser, S, top100_ranking, top_n=10):
    """
    Generate movie recommendations using Item-Based Collaborative Filtering.
    
    Args:
        newuser (Series): User's ratings for movies
        S (DataFrame): Similarity matrix
        top100_ranking (Series): Top 100 popular movies for fallback
        top_n (int): Number of recommendations to return
    
    Returns:
        list: Top N recommended movie IDs
    """
    # Validate ratings are between 1 and 5
    newuser = newuser.where((newuser >= 1) & (newuser <= 5))
    
    # Initialize predictions
    predictions = newuser.copy()
    
    # Calculate predictions for unrated movies
    for movie in newuser.index[newuser.isna()]:
        # Get similarities and ratings for rated movies
        similarities = S.loc[movie]
        rated_movies = newuser[newuser.notna()]
        
        # Calculate weighted sum of similarities
        weighted_sum = (similarities * rated_movies).sum()
        similarity_sum = similarities[rated_movies.index].sum()
        
        # Skip if no similar movies found
        if similarity_sum == 0:
            continue
            
        # Calculate prediction
        predictions[movie] = weighted_sum / similarity_sum
    
    # Get top N predictions from unrated movies
    top_predictions = (predictions[newuser.isna()]
                      .dropna()
                      .sort_values(ascending=False)
                      .head(top_n)
                      .index
                      .tolist())
    
    # If fewer than top_n predictions, add popular movies
    if len(top_predictions) < top_n:
        # Get movies not rated by user
        unrated = set(newuser[newuser.isna()].index)
        # Add popular movies until we have top_n recommendations
        for movie in top100_ranking:
            if movie in unrated and movie not in top_predictions:
                top_predictions.append(movie)
                if len(top_predictions) == top_n:
                    break
    
    return top_predictions

def main():
    """
    Main function to demonstrate the recommender system.
    """
    # Load data
    print("Loading data...")
    ratings, movies = load_data()
    
    # Get top 10 popular movies
    print("\nTop 10 Popular Movies:")
    top_10 = get_top_popular_movies(ratings, movies)
    print(top_10[['MovieID', 'Title', 'avg_rating', 'ratings_count']])
    
    # Generate top 100 movies for System II
    print("\nGenerating top 100 movies for System II...")
    top_100 = get_top_100_movies(ratings, movies)
    
    print("\nDone! The system is ready to provide recommendations.")

if __name__ == "__main__":
    main() 