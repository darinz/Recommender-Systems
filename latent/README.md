# Latent Factor Models for Recommender Systems

This directory contains implementations of latent factor models for recommender systems in both R and Python. These models capture the underlying structure of user-item interactions by learning low-dimensional representations of users and items.

## Overview

Latent factor models are a fundamental approach in collaborative filtering that:
- Learn low-dimensional representations (factors) for users and items
- Capture implicit preferences and characteristics
- Enable personalized recommendations based on learned patterns
- Handle sparse rating matrices effectively

## Files

### `latent_factor_models.R`
R implementation featuring:
- **SVD (Singular Value Decomposition)**: Matrix factorization using singular value decomposition
- **NMF (Non-negative Matrix Factorization)**: Factorization with non-negative constraints
- Synthetic data generation with latent structure
- Visualization of rating distributions and matrix heatmaps
- Model comparison with MAE and RMSE metrics

### `latent_factor_models.py`
Python implementation featuring:
- **LatentFactorModel**: Custom implementation with SGD optimization
- **SVDppModel**: Enhanced SVD++ algorithm with implicit feedback
- **NMF**: Non-negative matrix factorization using scikit-learn
- Comprehensive evaluation and visualization suite
- Advanced analysis including factor importance and bias distributions

## Key Features

### R Implementation
- Uses `recommenderlab` package for SVD implementation
- Implements NMF using the `NMF` package
- Generates synthetic data with controlled latent structure
- Provides basic visualization of results
- Includes train/test split for evaluation

### Python Implementation
- **Custom Latent Factor Model**: 
  - Stochastic Gradient Descent optimization
  - Configurable number of factors, learning rate, and regularization
  - User and item bias terms
  - Training history tracking
  
- **SVD++ Model**:
  - Enhanced version of SVD with implicit feedback
  - More sophisticated update rules
  - Better handling of missing data
  
- **Comprehensive Evaluation**:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Square Error)
  - Coverage metrics
  - Detailed visualizations

## Usage

### R Usage
```r
# Load the script
source("latent_factor_models.R")

# The script automatically:
# 1. Generates synthetic data
# 2. Trains SVD and NMF models
# 3. Evaluates performance
# 4. Creates visualizations
```

### Python Usage
```python
# Import the models
from latent_factor_models import LatentFactorModel, SVDppModel

# Create and train a model
model = LatentFactorModel(n_factors=10, learning_rate=0.01)
model.fit(ratings_df)

# Make predictions
prediction = model.predict(user_id, item_id)

# Get recommendations
recommendations = model.recommend(user_id, n_recommendations=5)

# Find similar items
similar_items = model.get_similar_items(item_id, n_similar=5)
```

## Model Parameters

### LatentFactorModel
- `n_factors`: Number of latent factors (default: 10)
- `learning_rate`: SGD learning rate (default: 0.01)
- `regularization`: L2 regularization parameter (default: 0.1)
- `n_epochs`: Number of training epochs (default: 100)
- `random_state`: Random seed for reproducibility (default: 42)

### SVDppModel
- Same parameters as LatentFactorModel
- Enhanced with implicit feedback handling

## Evaluation Metrics

### Accuracy Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual ratings
- **RMSE (Root Mean Square Error)**: Square root of average squared differences

### Coverage Metrics
- **Coverage**: Percentage of test items for which predictions can be made

## Visualizations

The Python implementation provides extensive visualizations:
- Training history curves
- User and item factor distributions
- Factor importance analysis
- Model comparison charts
- Prediction vs actual scatter plots
- Bias distributions
- Factor correlation matrices

## Key Insights

### Factor Analysis
- Factors capture different aspects of user preferences and item characteristics
- Factor importance is measured by variance across users/items
- Correlation analysis reveals relationships between factors

### Model Comparison
- SVD++ typically outperforms basic latent factor models
- NMF provides interpretable non-negative factors
- Different models excel at different aspects of recommendation

### Bias Analysis
- User biases capture overall rating tendencies
- Item biases reflect general popularity
- Global mean provides baseline for predictions

## Dependencies

### R Dependencies
- `recommenderlab`: For SVD implementation
- `ggplot2`: For visualizations
- `dplyr` and `tidyr`: For data manipulation
- `NMF`: For non-negative matrix factorization
- `gridExtra`: For plot arrangement

### Python Dependencies
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib` and `seaborn`: Visualizations
- `sklearn`: For NMF implementation and metrics

## Performance Considerations

- Training time scales with number of factors and epochs
- Memory usage depends on number of users and items
- SVD++ requires more computational resources than basic latent factor models
- NMF can be slower but provides more interpretable results

## Future Enhancements

Potential improvements include:
- Support for implicit feedback data
- Integration with deep learning approaches
- Real-time model updates
- Hyperparameter optimization
- Support for additional evaluation metrics
- Integration with streaming data sources 