# Collaborative Filtering

This folder contains implementations of collaborative filtering recommendation systems in both Python and R. Collaborative filtering is a technique used in recommendation systems that makes predictions about a user's interests by collecting preferences from many users.

## Overview

Collaborative filtering works by finding users or items that are similar to each other based on their rating patterns. There are two main approaches:

1. **User-based Collaborative Filtering (UBCF)**: Finds similar users and recommends items that similar users have rated highly
2. **Item-based Collaborative Filtering (IBCF)**: Finds similar items and recommends items similar to those the user has already rated highly

## Files

- `collaborative_filtering.py` - Python implementation with comprehensive features
- `collaborative_filtering.R` - R implementation using the recommenderlab package

## Python Implementation (`collaborative_filtering.py`)

### Features

The Python implementation provides a flexible `CollaborativeFiltering` class with the following capabilities:

- **Multiple similarity metrics**:
  - Cosine similarity
  - Pearson correlation
  - Jaccard similarity
  - Adjusted cosine similarity

- **Both user-based and item-based approaches**
- **Configurable number of neighbors** for recommendations
- **Comprehensive evaluation and visualization**

### Usage

```python
from collaborative_filtering import CollaborativeFiltering
import pandas as pd

# Load your ratings data
ratings_df = pd.DataFrame({
    'user_id': [...],
    'item_id': [...],
    'rating': [...]
})

# User-based collaborative filtering
user_cf = CollaborativeFiltering(
    method='user',
    similarity_metric='cosine',
    k_neighbors=10
)
user_cf.fit(ratings_df)

# Get recommendations
recommendations = user_cf.recommend(user_id=1, n_recommendations=5)

# Item-based collaborative filtering
item_cf = CollaborativeFiltering(
    method='item',
    similarity_metric='pearson',
    k_neighbors=10
)
item_cf.fit(ratings_df)

# Get similar items
similar_items = item_cf.get_similar_items(item_id=1, n_similar=5)
```

### Key Methods

- `fit(ratings_df)`: Train the model on rating data
- `predict(user_id, item_id)`: Predict rating for a specific user-item pair
- `recommend(user_id, n_recommendations)`: Get top recommendations for a user
- `get_similar_users(user_id, n_similar)`: Find most similar users
- `get_similar_items(item_id, n_similar)`: Find most similar items

### Visualization

The implementation includes comprehensive visualization capabilities:

- Rating matrix heatmap
- User similarity matrix
- Item similarity matrix
- Rating distribution
- Method comparison plots
- Similarity distribution analysis

## R Implementation (`collaborative_filtering.R`)

### Features

The R implementation uses the `recommenderlab` package and provides:

- **Built-in UBCF and IBCF methods**
- **Automatic model training and evaluation**
- **Visualization with ggplot2**
- **Synthetic data generation for testing**

### Usage

```r
library(recommenderlab)

# Load and prepare data
rating_matrix <- as(rating_data, "realRatingMatrix")

# Train user-based collaborative filtering
ubcf_model <- Recommender(rating_matrix, method = "UBCF")

# Train item-based collaborative filtering
ibcf_model <- Recommender(rating_matrix, method = "IBCF")

# Generate recommendations
recommendations <- predict(ubcf_model, rating_matrix[1:5], n = 5)
```

## Performance Comparison

The Python implementation includes a comprehensive performance comparison that evaluates:

- Mean predicted ratings
- Standard deviation of predictions
- Range of predicted ratings
- Similarity score distributions

## Requirements

### Python Dependencies
```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```

### R Dependencies
```
recommenderlab
ggplot2
dplyr
tidyr
gridExtra
```

## Key Differences

| Feature | Python Implementation | R Implementation |
|---------|---------------------|------------------|
| Similarity Metrics | 4 options (cosine, pearson, jaccard, adjusted_cosine) | Built-in methods |
| Customization | Highly configurable | Limited to recommenderlab options |
| Visualization | Comprehensive plots | Basic ggplot2 plots |
| Performance Analysis | Detailed comparison | Basic evaluation |
| Code Structure | Object-oriented class | Functional approach |

## Use Cases

- **E-commerce**: Product recommendations based on user behavior
- **Streaming Services**: Movie/TV show recommendations
- **Social Media**: Content recommendations
- **Academic Research**: Recommendation system development and evaluation

## Best Practices

1. **Data Preprocessing**: Handle missing values and normalize ratings appropriately
2. **Similarity Metric Selection**: Choose based on your data characteristics
3. **Neighbor Selection**: Balance between accuracy and computational cost
4. **Evaluation**: Use cross-validation and multiple metrics
5. **Scalability**: Consider matrix factorization for large datasets

## Future Enhancements

- Matrix factorization methods (SVD, NMF)
- Deep learning approaches
- Real-time recommendation updates
- A/B testing framework
- Cold-start problem solutions 