# Latent Factor Models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings('ignore')

class LatentFactorModel:
    """Basic Latent Factor Model with SGD optimization"""
    
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.1, 
                 n_epochs=100, random_state=42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        self.training_history = []
        
    def fit(self, ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        """Fit the latent factor model"""
        # Create user and item mappings
        self.user_mapping = {user: idx for idx, user in enumerate(ratings_df[user_col].unique())}
        self.item_mapping = {item: idx for idx, item in enumerate(ratings_df[item_col].unique())}
        
        self.n_users = len(self.user_mapping)
        self.n_items = len(self.item_mapping)
        
        # Initialize factors and biases
        np.random.seed(self.random_state)
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        
        # Compute global mean
        self.global_mean = ratings_df[rating_col].mean()
        
        # Convert to numpy arrays for faster computation
        user_indices = np.array([self.user_mapping[user] for user in ratings_df[user_col]])
        item_indices = np.array([self.item_mapping[item] for item in ratings_df[item_col]])
        ratings = np.array(ratings_df[rating_col])
        
        # SGD training
        for epoch in range(self.n_epochs):
            total_error = 0
            
            # Shuffle the data
            indices = np.random.permutation(len(ratings))
            
            for idx in indices:
                u = user_indices[idx]
                i = item_indices[idx]
                r = ratings[idx]
                
                # Predict rating
                pred = self._predict_single(u, i)
                
                # Compute error
                error = r - pred
                total_error += error ** 2
                
                # Update factors and biases
                self._update_factors(u, i, error)
            
            # Store training history
            avg_error = total_error / len(ratings)
            self.training_history.append(avg_error)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Average Error = {avg_error:.4f}")
        
        return self
    
    def _predict_single(self, user_idx, item_idx):
        """Predict rating for a single user-item pair"""
        return (self.global_mean + 
                self.user_biases[user_idx] + 
                self.item_biases[item_idx] + 
                np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
    
    def _update_factors(self, user_idx, item_idx, error):
        """Update factors and biases using SGD"""
        # Update user factors
        self.user_factors[user_idx] += (self.learning_rate * 
                                       (error * self.item_factors[item_idx] - 
                                        self.regularization * self.user_factors[user_idx]))
        
        # Update item factors
        self.item_factors[item_idx] += (self.learning_rate * 
                                       (error * self.user_factors[user_idx] - 
                                        self.regularization * self.item_factors[item_idx]))
        
        # Update biases
        self.user_biases[user_idx] += self.learning_rate * (error - self.regularization * self.user_biases[user_idx])
        self.item_biases[item_idx] += self.learning_rate * (error - self.regularization * self.item_biases[item_idx])
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        return self._predict_single(user_idx, item_idx)
    
    def recommend(self, user_id, n_recommendations=5, exclude_rated=True):
        """Generate top-n recommendations for a user"""
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        user_factor = self.user_factors[user_idx]
        
        # Predict ratings for all items
        predictions = []
        for item_id, item_idx in self.item_mapping.items():
            if exclude_rated:
                # Skip if user has rated this item (would need to track rated items)
                pass
            
            pred_rating = self._predict_single(user_idx, item_idx)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_similar_items(self, item_id, n_similar=5):
        """Find items similar to the given item based on latent factors"""
        if item_id not in self.item_mapping:
            return []
        
        item_idx = self.item_mapping[item_id]
        item_factor = self.item_factors[item_idx]
        
        # Compute similarities with all other items
        similarities = []
        for other_item_id, other_item_idx in self.item_mapping.items():
            if other_item_id != item_id:
                other_factor = self.item_factors[other_item_idx]
                similarity = np.dot(item_factor, other_factor) / (
                    np.linalg.norm(item_factor) * np.linalg.norm(other_factor)
                )
                similarities.append((other_item_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]

class SVDppModel:
    """SVD++ Model with implicit feedback"""
    
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.1, 
                 n_epochs=100, random_state=42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.implicit_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        self.user_items = None
        
    def fit(self, ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        """Fit the SVD++ model"""
        # Create mappings
        self.user_mapping = {user: idx for idx, user in enumerate(ratings_df[user_col].unique())}
        self.item_mapping = {item: idx for idx, item in enumerate(ratings_df[item_col].unique())}
        
        self.n_users = len(self.user_mapping)
        self.n_items = len(self.item_mapping)
        
        # Initialize factors
        np.random.seed(self.random_state)
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.implicit_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        
        # Compute global mean
        self.global_mean = ratings_df[rating_col].mean()
        
        # Create user-item mapping for implicit feedback
        self.user_items = {}
        for user_id in ratings_df[user_col].unique():
            user_idx = self.user_mapping[user_id]
            user_ratings = ratings_df[ratings_df[user_col] == user_id]
            self.user_items[user_idx] = [self.item_mapping[item] for item in user_ratings[item_col]]
        
        # Convert to numpy arrays
        user_indices = np.array([self.user_mapping[user] for user in ratings_df[user_col]])
        item_indices = np.array([self.item_mapping[item] for item in ratings_df[item_col]])
        ratings = np.array(ratings_df[rating_col])
        
        # SGD training
        for epoch in range(self.n_epochs):
            total_error = 0
            
            # Shuffle the data
            indices = np.random.permutation(len(ratings))
            
            for idx in indices:
                u = user_indices[idx]
                i = item_indices[idx]
                r = ratings[idx]
                
                # Predict rating
                pred = self._predict_single(u, i)
                
                # Compute error
                error = r - pred
                total_error += error ** 2
                
                # Update factors
                self._update_factors(u, i, error)
            
            if epoch % 20 == 0:
                avg_error = total_error / len(ratings)
                print(f"Epoch {epoch}: Average Error = {avg_error:.4f}")
        
        return self
    
    def _predict_single(self, user_idx, item_idx):
        """Predict rating for a single user-item pair"""
        # Basic prediction
        pred = (self.global_mean + 
                self.user_biases[user_idx] + 
                self.item_biases[item_idx] + 
                np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        
        # Add implicit feedback term
        if user_idx in self.user_items:
            user_rated_items = self.user_items[user_idx]
            if len(user_rated_items) > 0:
                implicit_sum = np.sum(self.implicit_factors[user_rated_items], axis=0)
                pred += np.dot(self.user_factors[user_idx], implicit_sum) / np.sqrt(len(user_rated_items))
        
        return pred
    
    def _update_factors(self, user_idx, item_idx, error):
        """Update factors using SGD"""
        # Update user factors
        self.user_factors[user_idx] += (self.learning_rate * 
                                       (error * self.item_factors[item_idx] - 
                                        self.regularization * self.user_factors[user_idx]))
        
        # Update item factors
        self.item_factors[item_idx] += (self.learning_rate * 
                                       (error * self.user_factors[user_idx] - 
                                        self.regularization * self.item_factors[item_idx]))
        
        # Update biases
        self.user_biases[user_idx] += self.learning_rate * (error - self.regularization * self.user_biases[user_idx])
        self.item_biases[item_idx] += self.learning_rate * (error - self.regularization * self.item_biases[item_idx])
        
        # Update implicit factors
        if user_idx in self.user_items:
            user_rated_items = self.user_items[user_idx]
            if len(user_rated_items) > 0:
                factor_update = (error * self.user_factors[user_idx] / np.sqrt(len(user_rated_items)) - 
                               self.regularization * self.implicit_factors[item_idx])
                self.implicit_factors[item_idx] += self.learning_rate * factor_update
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        return self._predict_single(user_idx, item_idx)

# Generate synthetic data with latent structure
np.random.seed(42)
n_users = 300
n_items = 200
n_ratings = 3000

# Create synthetic ratings with latent factors
ratings_data = []
for user_id in range(n_users):
    n_user_ratings = np.random.randint(8, 25)
    rated_items = np.random.choice(n_items, n_user_ratings, replace=False)
    
    for item_id in rated_items:
        # Create latent factor structure
        # Factor 1: Action vs Drama preference
        # Factor 2: Complexity preference
        # Factor 3: Genre preference
        
        user_action_pref = np.random.normal(0, 1)  # User's action preference
        user_complexity_pref = np.random.normal(0, 1)  # User's complexity preference
        user_genre_pref = np.random.normal(0, 1)  # User's genre preference
        
        item_action_level = np.random.normal(0, 1)  # Item's action level
        item_complexity = np.random.normal(0, 1)  # Item's complexity
        item_genre = np.random.normal(0, 1)  # Item's genre
        
        # Compute rating based on latent factors
        latent_score = (user_action_pref * item_action_level + 
                       user_complexity_pref * item_complexity + 
                       user_genre_pref * item_genre)
        
        # Add noise and convert to 1-5 scale
        rating = max(1, min(5, 3 + latent_score + np.random.normal(0, 0.5)))
        ratings_data.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating
        })

ratings_df = pd.DataFrame(ratings_data)

print("Synthetic Dataset with Latent Structure:")
print(f"Number of users: {n_users}")
print(f"Number of items: {n_items}")
print(f"Number of ratings: {len(ratings_df)}")
print(f"Sparsity: {1 - len(ratings_df) / (n_users * n_items):.3f}")

# Split data for evaluation
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Train different latent factor models
print("\n=== Training Latent Factor Models ===")

# Basic Latent Factor Model
lf_model = LatentFactorModel(n_factors=10, learning_rate=0.01, regularization=0.1, n_epochs=100)
lf_model.fit(train_df)

# SVD++ Model
svdpp_model = SVDppModel(n_factors=10, learning_rate=0.01, regularization=0.1, n_epochs=100)
svdpp_model.fit(train_df)

# NMF Model (using sklearn)
train_matrix = train_df.pivot_table(
    index='user_id', columns='item_id', values='rating', fill_value=0
)
nmf_model = NMF(n_components=10, random_state=42, max_iter=100)
nmf_user_factors = nmf_model.fit_transform(train_matrix)
nmf_item_factors = nmf_model.components_.T

# Evaluate models
def evaluate_model(model, test_df, model_type='custom'):
    """Evaluate model on test set"""
    predictions = []
    actuals = []
    
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        actual_rating = row['rating']
        
        if model_type == 'nmf':
            # For NMF, need to handle missing users/items
            if user_id in train_matrix.index and item_id in train_matrix.columns:
                user_idx = train_matrix.index.get_loc(user_id)
                item_idx = train_matrix.columns.get_loc(item_id)
                pred_rating = np.dot(nmf_user_factors[user_idx], nmf_item_factors[item_idx])
            else:
                pred_rating = np.nan
        else:
            pred_rating = model.predict(user_id, item_id)
        
        if not np.isnan(pred_rating):
            predictions.append(pred_rating)
            actuals.append(actual_rating)
    
    if len(predictions) == 0:
        return {'mae': np.inf, 'rmse': np.inf, 'coverage': 0}
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    coverage = len(predictions) / len(test_df)
    
    return {'mae': mae, 'rmse': rmse, 'coverage': coverage}

# Evaluate all models
models = {
    'Latent Factor': lf_model,
    'SVD++': svdpp_model,
    'NMF': None
}

results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    model_type = 'nmf' if name == 'NMF' else 'custom'
    results[name] = evaluate_model(model, test_df, model_type)

# Display results
print("\n=== Evaluation Results ===")
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Coverage: {metrics['coverage']:.4f}")
    print()

# Visualization
plt.figure(figsize=(20, 12))

# Plot 1: Training history
plt.subplot(3, 4, 1)
plt.plot(lf_model.training_history, label='Latent Factor')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Average Error')
plt.legend()

# Plot 2: User factors visualization (first 2 dimensions)
plt.subplot(3, 4, 2)
user_factors_2d = lf_model.user_factors[:, :2]
plt.scatter(user_factors_2d[:, 0], user_factors_2d[:, 1], alpha=0.6)
plt.title('User Factors (First 2 Dimensions)')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')

# Plot 3: Item factors visualization (first 2 dimensions)
plt.subplot(3, 4, 3)
item_factors_2d = lf_model.item_factors[:, :2]
plt.scatter(item_factors_2d[:, 0], item_factors_2d[:, 1], alpha=0.6)
plt.title('Item Factors (First 2 Dimensions)')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')

# Plot 4: Factor importance
plt.subplot(3, 4, 4)
factor_importance = np.var(lf_model.user_factors, axis=0)
plt.bar(range(len(factor_importance)), factor_importance)
plt.title('Factor Importance (Variance)')
plt.xlabel('Factor')
plt.ylabel('Variance')

# Plot 5: MAE comparison
plt.subplot(3, 4, 5)
mae_values = [results[name]['mae'] for name in results.keys()]
plt.bar(results.keys(), mae_values, color=['blue', 'red', 'green'])
plt.title('MAE Comparison')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)

# Plot 6: RMSE comparison
plt.subplot(3, 4, 6)
rmse_values = [results[name]['rmse'] for name in results.keys()]
plt.bar(results.keys(), rmse_values, color=['blue', 'red', 'green'])
plt.title('RMSE Comparison')
plt.ylabel('Root Mean Square Error')
plt.xticks(rotation=45)

# Plot 7: Coverage comparison
plt.subplot(3, 4, 7)
coverage_values = [results[name]['coverage'] for name in results.keys()]
plt.bar(results.keys(), coverage_values, color=['blue', 'red', 'green'])
plt.title('Coverage Comparison')
plt.ylabel('Coverage')
plt.xticks(rotation=45)

# Plot 8: Prediction vs Actual (Latent Factor)
plt.subplot(3, 4, 8)
lf_predictions = []
lf_actuals = []
for _, row in test_df.head(100).iterrows():
    pred = lf_model.predict(row['user_id'], row['item_id'])
    if not np.isnan(pred):
        lf_predictions.append(pred)
        lf_actuals.append(row['rating'])

plt.scatter(lf_actuals, lf_predictions, alpha=0.6)
plt.plot([1, 5], [1, 5], 'r--', alpha=0.8)
plt.title('Latent Factor: Predicted vs Actual')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')

# Plot 9: User bias distribution
plt.subplot(3, 4, 9)
plt.hist(lf_model.user_biases, bins=30, alpha=0.7, edgecolor='black')
plt.title('User Bias Distribution')
plt.xlabel('Bias')
plt.ylabel('Frequency')

# Plot 10: Item bias distribution
plt.subplot(3, 4, 10)
plt.hist(lf_model.item_biases, bins=30, alpha=0.7, edgecolor='black')
plt.title('Item Bias Distribution')
plt.xlabel('Bias')
plt.ylabel('Frequency')

# Plot 11: Factor correlation matrix
plt.subplot(3, 4, 11)
factor_corr = np.corrcoef(lf_model.user_factors.T)
sns.heatmap(factor_corr, cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation'})
plt.title('Factor Correlation Matrix')

# Plot 12: Model comparison summary
plt.subplot(3, 4, 12)
comparison_metrics = ['MAE', 'RMSE', 'Coverage']
comparison_values = [
    [results[name]['mae'] for name in results.keys()],
    [results[name]['rmse'] for name in results.keys()],
    [results[name]['coverage'] for name in results.keys()]
]

x = np.arange(len(comparison_metrics))
width = 0.25

for i, (name, values) in enumerate(zip(results.keys(), np.array(comparison_values).T)):
    plt.bar(x + i*width, values, width, label=name)

plt.title('Model Comparison')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.xticks(x + width, comparison_metrics)
plt.legend()

plt.tight_layout()
plt.show()

# Detailed analysis
print("\n=== Detailed Analysis ===")

# Compare factor interpretations
print("Factor Analysis:")
for i in range(min(5, lf_model.n_factors)):
    user_factor_std = np.std(lf_model.user_factors[:, i])
    item_factor_std = np.std(lf_model.item_factors[:, i])
    print(f"Factor {i+1}: User std = {user_factor_std:.3f}, Item std = {item_factor_std:.3f}")

# Compare prediction patterns
test_sample = test_df.head(50)
lf_preds = []
svdpp_preds = []
actuals = []

for _, row in test_sample.iterrows():
    lf_pred = lf_model.predict(row['user_id'], row['item_id'])
    svdpp_pred = svdpp_model.predict(row['user_id'], row['item_id'])
    
    if not (np.isnan(lf_pred) or np.isnan(svdpp_pred)):
        lf_preds.append(lf_pred)
        svdpp_preds.append(svdpp_pred)
        actuals.append(row['rating'])

print(f"\nPrediction Statistics:")
print(f"Latent Factor:")
print(f"  Mean: {np.mean(lf_preds):.3f}")
print(f"  Std: {np.std(lf_preds):.3f}")
print(f"  Range: [{np.min(lf_preds):.3f}, {np.max(lf_preds):.3f}]")

print(f"\nSVD++:")
print(f"  Mean: {np.mean(svdpp_preds):.3f}")
print(f"  Std: {np.std(svdpp_preds):.3f}")
print(f"  Range: [{np.min(svdpp_preds):.3f}, {np.max(svdpp_preds):.3f}]")

# Test recommendations
test_user = 0
print(f"\nTop 5 recommendations for User {test_user}:")
recommendations = lf_model.recommend(test_user, n_recommendations=5)
for i, (item_id, pred_rating) in enumerate(recommendations):
    print(f"  {i+1}. Item {item_id}: Predicted rating = {pred_rating:.3f}")

# Test similar items
test_item = 0
print(f"\nTop 5 similar items to Item {test_item}:")
similar_items = lf_model.get_similar_items(test_item, n_similar=5)
for i, (item_id, similarity) in enumerate(similar_items):
    print(f"  {i+1}. Item {item_id}: Similarity = {similarity:.3f}")