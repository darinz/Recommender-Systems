# Content-Based Movie Recommender System

## Overview

This project implements a **content-based movie recommendation system** that demonstrates key concepts in recommender systems. The system analyzes movie features (genre, year, director, etc.) and user preferences to generate personalized movie recommendations.

## Key Features

- **Content-Based Filtering**: Analyzes movie features to find similar items
- **Feature Engineering**: Extracts and processes movie attributes
- **User Profile Creation**: Builds user preferences from rating history
- **Similarity Computation**: Multiple similarity metrics (cosine, euclidean, pearson)
- **Visualization**: Interactive plots showing recommendations and user preferences
- **Evaluation**: Metrics to assess recommendation quality
- **Educational**: Comprehensive comments explaining each step

## Learning Objectives

This system helps you understand:

1. **Content-Based Filtering**: How to recommend items based on their features
2. **Feature Engineering**: Processing and combining different types of features
3. **User Profiling**: Creating user preference models from behavior data
4. **Similarity Metrics**: Different ways to measure item similarity
5. **Recommendation Evaluation**: How to assess recommendation quality
6. **Data Preprocessing**: Cleaning and preparing data for ML algorithms

## Architecture

```
Movie Data → Feature Engineering → Item Profiles → User Profiles → Recommendations
     ↓              ↓                ↓              ↓              ↓
  Raw Movies → Extract Features → Create Vectors → Build Preferences → Generate Recs
```

## Installation

1. **Install Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python3 demo_movie_recommender.py
   ```

## Usage

### Quick Start
```bash
# Run the full demonstration
python3 movie_recommender.py

# Or run the demo version
python3 demo_movie_recommender.py
```

### Step-by-Step Usage
```python
from movie_recommender import ContentBasedRecommender

# Initialize recommender
recommender = ContentBasedRecommender(similarity_metric='cosine')

# Create item profiles
item_profiles = recommender.create_item_profiles(movies_df, feature_columns, text_columns)

# Create user profiles
user_profiles = recommender.create_user_profiles(ratings_df, movies_df)

# Generate recommendations
recommendations = recommender.recommend(user_id=1, n_recommendations=10)
```

## Features Explained

### 1. Content-Based Filtering
- **What it does**: Recommends movies similar to what the user has liked before
- **How it works**: Analyzes movie features (genre, year, director, etc.)
- **Example**: If a user likes action movies from the 2000s, recommend similar action movies

### 2. Feature Engineering
- **Numerical Features**: Year, rating, budget
- **Categorical Features**: Genre, director (encoded as numerical)
- **Text Features**: Movie descriptions (processed with TF-IDF)

### 3. User Profiling
- **Process**: Analyzes user's rating history
- **Output**: Creates a "taste profile" for each user
- **Weighting**: Recent and higher-rated movies have more influence

### 4. Similarity Metrics
- **Cosine Similarity**: Measures angle between feature vectors
- **Euclidean Distance**: Measures straight-line distance
- **Pearson Correlation**: Measures linear relationship

## Evaluation Metrics

- **Mean Squared Error (MSE)**: Average squared difference between predicted and actual ratings
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual ratings
- **Coverage**: Percentage of items that can be recommended
- **Diversity**: How different the recommended items are from each other

## Visualizations

The system provides several visualizations:

1. **Recommendation Similarities**: Bar chart showing similarity scores
2. **Feature Importance**: Which features matter most for each user
3. **User-Item Profiles**: 2D visualization of user and item profiles
4. **Similarity Distributions**: Histograms of similarity scores

## Customization

### Adding New Features
```python
# Add new feature columns
feature_columns = ['year', 'rating', 'budget', 'new_feature']
```

### Changing Similarity Metric
```python
# Use different similarity metrics
recommender = ContentBasedRecommender(similarity_metric='euclidean')
```

### Adjusting Parameters
```python
# Modify TF-IDF parameters
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
```

## Code Structure

```
content/
├── movie_recommender.py          # Main recommender system
├── content-based_recommender.py  # Core recommender class
├── demo_movie_recommender.py     # Demo script
├── requirements.txt              # Dependencies
└── README.md                    # This file
```

## Educational Content

### Key Concepts Covered:

1. **Content-Based Filtering**
   - How to represent items as feature vectors
   - How to compute similarity between items
   - How to generate recommendations

2. **Feature Engineering**
   - Handling different data types (numerical, categorical, text)
   - Feature normalization and encoding
   - Dimensionality reduction techniques

3. **User Modeling**
   - Creating user profiles from behavior data
   - Weighting strategies for user preferences
   - Cold-start problem handling

4. **Evaluation**
   - Offline evaluation metrics
   - Train/test splitting strategies
   - Performance analysis

## Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure all dependencies are installed
2. **Memory Issues**: Reduce dataset size or feature count
3. **Poor Recommendations**: Try different similarity metrics or feature combinations
4. **Visualization Errors**: Check matplotlib backend settings

### Performance Tips:

1. **Use smaller datasets** for faster experimentation
2. **Reduce feature count** if memory is limited
3. **Cache results** for repeated computations
4. **Use efficient similarity metrics** for large datasets

## Further Reading

- [Recommender Systems Handbook](https://link.springer.com/book/10.1007/978-1-4899-7637-6)
- [Content-Based Filtering](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

## Contributing

Feel free to contribute by:
- Adding new similarity metrics
- Improving the visualization
- Adding more evaluation metrics
- Enhancing the documentation