# Deep Recommender Systems

This folder contains implementations of deep learning-based recommender systems in both Python (PyTorch) and R (Keras). The implementations demonstrate state-of-the-art deep learning approaches for collaborative filtering and rating prediction.

## Overview

Deep recommender systems leverage neural networks to capture complex, non-linear patterns in user-item interactions that traditional collaborative filtering methods might miss. This implementation includes three popular deep learning architectures:

1. **Neural Collaborative Filtering (NCF)** - Combines matrix factorization with multi-layer perceptrons
2. **Wide & Deep** - Combines linear models with deep neural networks
3. **Neural Matrix Factorization (NeuMF)** - Integrates generalized matrix factorization with neural networks

## Files

- `deep_rec_sys.py` - Python implementation using PyTorch (525 lines)
- `deep_rec_sys.R` - R implementation using Keras (167 lines)

## Features

### Synthetic Data Generation
Both implementations generate synthetic rating data with non-linear patterns to demonstrate the models' ability to capture complex interactions:

- **500 users** and **300 items**
- **3000 ratings** with non-linear interaction patterns
- Synthetic patterns using trigonometric functions and noise

### Model Architectures

#### 1. Neural Collaborative Filtering (NCF)
- User and item embeddings
- Multi-layer perceptron for learning non-linear interactions
- Dropout regularization for preventing overfitting

#### 2. Wide & Deep
- **Wide component**: Linear model for memorization
- **Deep component**: Neural network for generalization
- Combination of both approaches for better performance

#### 3. Neural Matrix Factorization (NeuMF)
- **GMF component**: Generalized matrix factorization
- **MLP component**: Multi-layer perceptron
- Integration of both components for enhanced performance

### Training and Evaluation

#### Python Implementation
- PyTorch-based training with customizable epochs
- Adam optimizer with learning rate scheduling
- Comprehensive evaluation metrics (MAE, RMSE)
- Training/validation loss tracking
- GPU support when available

#### R Implementation
- Keras-based implementation
- Adam optimizer with MSE loss
- Validation split during training
- Visualization of training history and predictions

### Visualization

Both implementations include comprehensive visualization capabilities:

- Training curves (loss vs. epochs)
- Model performance comparison
- Predicted vs. actual ratings
- Rating distribution analysis
- Model architecture comparisons

## Usage

### Python Implementation

```python
# Run the complete deep recommender system
python deep_rec_sys.py
```

The Python script will:
1. Generate synthetic data with non-linear patterns
2. Train all three models (NCF, Wide&Deep, NeuMF)
3. Evaluate performance on test set
4. Generate comprehensive visualizations

### R Implementation

```r
# Run the R implementation
Rscript deep_rec_sys.R
```

The R script will:
1. Generate synthetic rating data
2. Train the NCF model
3. Evaluate performance
4. Create visualizations using ggplot2

## Key Features

### Data Processing
- Automatic user/item ID mapping
- Train/validation/test splits
- Batch processing for efficient training
- Handling of sparse rating matrices

### Model Training
- Early stopping to prevent overfitting
- Learning rate scheduling
- Dropout regularization
- Weight initialization strategies

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Training/validation loss curves
- Prediction accuracy analysis

## Requirements

### Python Dependencies
```
torch
numpy
pandas
matplotlib
seaborn
scikit-learn
```

### R Dependencies
```
keras
tensorflow
ggplot2
dplyr
tidyr
gridExtra
```

## Performance Characteristics

The implementations demonstrate:

- **Non-linear pattern learning**: Models can capture complex user-item interactions
- **Scalability**: Efficient training on synthetic datasets
- **Regularization**: Dropout and early stopping prevent overfitting
- **Interpretability**: Visualization tools for model analysis

## Applications

These deep recommender systems are suitable for:

- Movie recommendation systems
- E-commerce product recommendations
- Music streaming services
- Social media content recommendations
- Any domain with user-item rating data

## Research Context

The implementations are based on seminal papers in deep recommender systems:

- **NCF**: "Neural Collaborative Filtering" (He et al., 2017)
- **Wide & Deep**: "Wide & Deep Learning for Recommender Systems" (Cheng et al., 2016)
- **NeuMF**: "Neural Collaborative Filtering" (He et al., 2017)

## Future Extensions

Potential enhancements could include:

- Attention mechanisms
- Graph neural networks
- Multi-modal recommendations
- Real-time training capabilities
- Production deployment considerations

## Notes

- Both implementations use synthetic data for demonstration
- Models are designed for rating prediction tasks
- GPU acceleration is supported in the Python version
- The R implementation focuses on NCF for simplicity
- All models include comprehensive evaluation and visualization 