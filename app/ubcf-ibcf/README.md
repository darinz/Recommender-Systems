# UBCF vs IBCF Comparison

This folder contains implementations and comparisons of User-Based Collaborative Filtering (UBCF) and Item-Based Collaborative Filtering (IBCF) recommendation algorithms. Both Python and R implementations are provided with comprehensive evaluation and visualization capabilities.

## Overview

Collaborative filtering is a fundamental approach in recommendation systems that can be implemented in two main ways:

- **User-Based Collaborative Filtering (UBCF)**: Finds similar users and recommends items that similar users have liked
- **Item-Based Collaborative Filtering (IBCF)**: Finds similar items and recommends items similar to what the user has already rated positively

## Files

### `ubcf_vs_ibcf.py`
A comprehensive Python implementation that includes:

- **UBCF Class**: Implements user-based collaborative filtering with configurable similarity metrics (Pearson correlation, cosine similarity)
- **IBCF Class**: Implements item-based collaborative filtering with adjusted cosine similarity and Pearson correlation
- **Evaluation Framework**: Comprehensive evaluation using MAE, RMSE, and coverage metrics
- **Visualization Suite**: 12 different plots including:
  - Rating matrix heatmaps
  - User/item similarity matrices
  - Performance comparison charts
  - Prediction vs actual scatter plots
  - Similarity distributions
  - Computational complexity analysis

### `ubcf_vs_ibcf.R`
An R implementation using the `recommenderlab` package that provides:

- **Synthetic Data Generation**: Creates clustered rating data for testing
- **Multiple Methods**: Tests both UBCF and IBCF with different similarity metrics
- **Evaluation Metrics**: MAE, RMSE, and coverage calculations
- **Visualization**: Rating distributions, heatmaps, and performance comparisons

## Key Features

### Similarity Metrics
- **Pearson Correlation**: Measures linear correlation between users/items
- **Cosine Similarity**: Measures angle between rating vectors
- **Adjusted Cosine**: Centers ratings before computing cosine similarity (IBCF)

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual ratings
- **RMSE (Root Mean Square Error)**: Square root of average squared differences
- **Coverage**: Percentage of test cases for which predictions can be made

### Performance Analysis
- **Computational Complexity**: UBCF (O(n²m)) vs IBCF (O(m²n))
- **Scalability**: Trade-offs between user and item-based approaches
- **Cold Start Handling**: How each method handles new users/items

## Usage

### Python Implementation

```python
# Import and create models
from ubcf_vs_ibcf import UBCF, IBCF

# Create UBCF model with Pearson correlation
ubcf_model = UBCF(similarity_metric='pearson', k_neighbors=10)
ubcf_model.fit(ratings_df)

# Create IBCF model with adjusted cosine
ibcf_model = IBCF(similarity_metric='adjusted_cosine', k_neighbors=10)
ibcf_model.fit(ratings_df)

# Make predictions
prediction = ubcf_model.predict(user_id, item_id)
```

### R Implementation

```r
# Load the script
source("ubcf_vs_ibcf.R")

# The script automatically:
# 1. Generates synthetic data
# 2. Trains UBCF and IBCF models
# 3. Evaluates performance
# 4. Creates visualizations
```

## Key Differences

| Aspect | UBCF | IBCF |
|--------|------|------|
| **Focus** | User similarities | Item similarities |
| **Scalability** | Better for fewer users | Better for fewer items |
| **Cold Start** | Struggles with new users | Struggles with new items |
| **Stability** | User preferences change | Item characteristics stable |
| **Memory** | User similarity matrix | Item similarity matrix |

## Requirements

### Python
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

### R
- recommenderlab
- ggplot2
- dplyr
- tidyr
- gridExtra

## Output

Both implementations generate comprehensive visualizations and analysis including:

1. **Performance Metrics**: MAE, RMSE, and coverage comparisons
2. **Similarity Analysis**: Distribution of user/item similarities
3. **Prediction Quality**: Predicted vs actual rating scatter plots
4. **Matrix Visualizations**: Heatmaps of rating and similarity matrices
5. **Statistical Analysis**: Detailed comparison of prediction patterns

## Applications

This comparison is useful for:

- **Research**: Understanding trade-offs between UBCF and IBCF
- **Education**: Learning collaborative filtering fundamentals
- **System Design**: Choosing appropriate approach for specific use cases
- **Benchmarking**: Establishing baseline performance metrics

## References

- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms
- Herlocker, J. L., Konstan, J. A., Borchers, A., & Riedl, J. (1999). An algorithmic framework for performing collaborative filtering 