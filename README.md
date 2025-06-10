# Movie Recommender System

This is a Python implementation of a movie recommender system based on the MovieLens dataset. The system provides two types of recommendations:

1. **System I**: Recommendation based on popularity
2. **System II**: Recommendation based on Item-Based Collaborative Filtering (IBCF)

## Features

- Load and preprocess MovieLens dataset
- Calculate movie popularity based on number of ratings and average ratings
- Implement item-based collaborative filtering
- Generate personalized movie recommendations
- Support for handling missing ratings and fallback to popular movies

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script to see the top 10 most popular movies and generate the top 100 movies for System II:

```bash
python movie_recommender.py
```

### Using the Recommendation Functions

```python
from movie_recommender import load_data, get_top_popular_movies, myIBCF

# Load data
ratings, movies = load_data()

# Get top 10 popular movies
top_10 = get_top_popular_movies(ratings, movies)

# Get recommendations for a new user
# Example: User rated movie "m1613" with 5 stars and "m1755" with 4 stars
newuser = pd.Series(index=S.index, data=np.nan)
newuser['m1613'] = 5
newuser['m1755'] = 4

# Get recommendations
recommendations = myIBCF(newuser, S, top100_ranking)
```

## Data

The system uses the MovieLens dataset which contains:
- Approximately 1 million anonymous ratings
- 3,706 movies
- 6,040 users
- Ratings on a 5-star scale

## Implementation Details

### System I: Popularity-Based Recommendations
The popularity-based recommendation system uses a combination of two key metrics to identify the most popular movies:

1. **Number of Ratings**: 
   - Movies must have at least 1,000 ratings to be considered
   - This threshold ensures statistical reliability and broad appeal
   - Helps avoid bias from movies with few ratings

2. **Average Rating**:
   - Calculated as the mean of all ratings for each movie
   - Provides a measure of audience satisfaction
   - Movies are ranked by their average rating among those meeting the minimum review count

The system combines these metrics to emphasize both quantity (wide interest) and quality (viewer satisfaction). This approach ensures that recommended movies are both well-regarded and broadly recognized.

### System II: Item-Based Collaborative Filtering (IBCF)
The IBCF system implements a sophisticated recommendation algorithm that:

1. **Data Preprocessing**:
   - Centers the rating matrix by subtracting row means
   - Handles missing values appropriately
   - Normalizes ratings for better comparison

2. **Similarity Calculation**:
   - Uses transformed cosine similarity between movies
   - Considers only movies rated by at least 3 common users
   - Transforms similarity scores to range [0,1]
   - Keeps top 30 similarities for each movie

3. **Prediction Formula**:
   For each unrated movie i, the predicted rating is calculated as:

   $$prediction_i = Σ(S_{ij} * w_j) / Σ(S_{ij})$$

   where:
   - $S_{ij}$ is the similarity between movies $i$ and $j$
   - $w_j$ is the user's rating for movie $j$
   - Sums are taken over all movies $j$ that the user has rated

4. **Fallback Mechanism**:
   - If fewer than 10 predictions are available
   - Falls back to System I's popularity-based recommendations
   - Excludes movies already rated by the user
   - Ensures a complete set of 10 recommendations

## Performance Considerations

- The system is optimized to handle the full MovieLens dataset
- Similarity matrix is precomputed and stored for efficiency
- Top 100 popular movies are cached for quick fallback recommendations
- Memory-efficient implementation for large-scale deployment

## License

This project is licensed under the MIT License - see the LICENSE file for details. 