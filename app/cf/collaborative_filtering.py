import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFiltering:
    def __init__(self, method='user', similarity_metric='cosine', k_neighbors=10):
        """
        Collaborative Filtering Recommender
        
        Parameters:
        -----------
        method : str
            'user' for user-based CF, 'item' for item-based CF
        similarity_metric : str
            'cosine', 'pearson', 'jaccard', 'adjusted_cosine'
        k_neighbors : int
            Number of neighbors to consider
        """
        self.method = method
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.rating_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.user_means = None
        self.item_means = None
        
    def fit(self, ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        """Fit the collaborative filtering model"""
        # Create rating matrix
        self.rating_matrix = ratings_df.pivot_table(
            index=user_col, 
            columns=item_col, 
            values=rating_col, 
            fill_value=np.nan
        )
        
        # Compute means
        self.user_means = self.rating_matrix.mean(axis=1)
        self.item_means = self.rating_matrix.mean(axis=0)
        
        # Compute similarities
        if self.method == 'user':
            self.user_similarity = self._compute_user_similarity()
        else:
            self.item_similarity = self._compute_item_similarity()
            
        return self
    
    def _compute_user_similarity(self):
        """Compute user similarity matrix"""
        if self.similarity_metric == 'cosine':
            # Fill NaN with 0 for cosine similarity
            matrix_filled = self.rating_matrix.fillna(0)
            return cosine_similarity(matrix_filled)
        
        elif self.similarity_metric == 'pearson':
            # Compute Pearson correlation for each user pair
            n_users = len(self.rating_matrix)
            similarity_matrix = np.zeros((n_users, n_users))
            
            for i in range(n_users):
                for j in range(i+1, n_users):
                    # Get common rated items
                    user_i_ratings = self.rating_matrix.iloc[i]
                    user_j_ratings = self.rating_matrix.iloc[j]
                    
                    common_items = ~(user_i_ratings.isna() | user_j_ratings.isna())
                    
                    if common_items.sum() > 1:
                        corr, _ = pearsonr(
                            user_i_ratings[common_items], 
                            user_j_ratings[common_items]
                        )
                        similarity_matrix[i, j] = corr
                        similarity_matrix[j, i] = corr
                    else:
                        similarity_matrix[i, j] = 0
                        similarity_matrix[j, i] = 0
            
            return similarity_matrix
        
        elif self.similarity_metric == 'jaccard':
            # Convert to binary (rated/not rated)
            binary_matrix = ~self.rating_matrix.isna()
            return cosine_similarity(binary_matrix)
    
    def _compute_item_similarity(self):
        """Compute item similarity matrix"""
        if self.similarity_metric == 'cosine':
            # Fill NaN with 0 for cosine similarity
            matrix_filled = self.rating_matrix.fillna(0)
            return cosine_similarity(matrix_filled.T)
        
        elif self.similarity_metric == 'adjusted_cosine':
            # Center by user means
            centered_matrix = self.rating_matrix.sub(self.user_means, axis=0)
            # Fill NaN with 0
            centered_matrix = centered_matrix.fillna(0)
            return cosine_similarity(centered_matrix.T)
        
        elif self.similarity_metric == 'pearson':
            # Compute Pearson correlation for each item pair
            n_items = len(self.rating_matrix.columns)
            similarity_matrix = np.zeros((n_items, n_items))
            
            for i in range(n_items):
                for j in range(i+1, n_items):
                    # Get common users
                    item_i_ratings = self.rating_matrix.iloc[:, i]
                    item_j_ratings = self.rating_matrix.iloc[:, j]
                    
                    common_users = ~(item_i_ratings.isna() | item_j_ratings.isna())
                    
                    if common_users.sum() > 1:
                        corr, _ = pearsonr(
                            item_i_ratings[common_users], 
                            item_j_ratings[common_users]
                        )
                        similarity_matrix[i, j] = corr
                        similarity_matrix[j, i] = corr
                    else:
                        similarity_matrix[i, j] = 0
                        similarity_matrix[j, i] = 0
            
            return similarity_matrix
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if self.method == 'user':
            return self._predict_user_based(user_id, item_id)
        else:
            return self._predict_item_based(user_id, item_id)
    
    def _predict_user_based(self, user_id, item_id):
        """User-based prediction"""
        if user_id not in self.rating_matrix.index or item_id not in self.rating_matrix.columns:
            return self.user_means.mean()
        
        user_idx = self.rating_matrix.index.get_loc(user_id)
        item_idx = self.rating_matrix.columns.get_loc(item_id)
        
        # Get user similarities
        user_similarities = self.user_similarity[user_idx]
        
        # Find users who rated this item
        item_ratings = self.rating_matrix.iloc[:, item_idx]
        rated_users = ~item_ratings.isna()
        
        if not rated_users.any():
            return self.user_means.mean()
        
        # Get similarities and ratings for users who rated this item
        similarities = user_similarities[rated_users]
        ratings = item_ratings[rated_users]
        
        # Sort by similarity and take top-k
        sorted_indices = np.argsort(similarities)[::-1][:self.k_neighbors]
        
        if len(sorted_indices) == 0:
            return self.user_means.mean()
        
        top_similarities = similarities.iloc[sorted_indices]
        top_ratings = ratings.iloc[sorted_indices]
        
        # Weighted average
        weighted_sum = np.sum(top_similarities * top_ratings)
        total_similarity = np.sum(np.abs(top_similarities))
        
        if total_similarity == 0:
            return top_ratings.mean()
        
        return weighted_sum / total_similarity
    
    def _predict_item_based(self, user_id, item_id):
        """Item-based prediction"""
        if user_id not in self.rating_matrix.index or item_id not in self.rating_matrix.columns:
            return self.item_means.mean()
        
        user_idx = self.rating_matrix.index.get_loc(user_id)
        item_idx = self.rating_matrix.columns.get_loc(item_id)
        
        # Get item similarities
        item_similarities = self.item_similarity[item_idx]
        
        # Find items rated by this user
        user_ratings = self.rating_matrix.iloc[user_idx]
        rated_items = ~user_ratings.isna()
        
        if not rated_items.any():
            return self.item_means.mean()
        
        # Get similarities and ratings for items rated by this user
        similarities = item_similarities[rated_items]
        ratings = user_ratings[rated_items]
        
        # Sort by similarity and take top-k
        sorted_indices = np.argsort(similarities)[::-1][:self.k_neighbors]
        
        if len(sorted_indices) == 0:
            return self.item_means.mean()
        
        top_similarities = similarities.iloc[sorted_indices]
        top_ratings = ratings.iloc[sorted_indices]
        
        # Weighted average
        weighted_sum = np.sum(top_similarities * top_ratings)
        total_similarity = np.sum(np.abs(top_similarities))
        
        if total_similarity == 0:
            return top_ratings.mean()
        
        return weighted_sum / total_similarity
    
    def recommend(self, user_id, n_recommendations=5):
        """Generate top-n recommendations for a user"""
        if user_id not in self.rating_matrix.index:
            return []
        
        user_ratings = self.rating_matrix.loc[user_id]
        unrated_items = user_ratings.isna()
        
        if not unrated_items.any():
            return []
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in user_ratings[unrated_items].index:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_similar_users(self, user_id, n_similar=5):
        """Get most similar users"""
        if user_id not in self.rating_matrix.index:
            return []
        
        user_idx = self.rating_matrix.index.get_loc(user_id)
        similarities = self.user_similarity[user_idx]
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1][1:n_similar+1]  # Exclude self
        similar_users = []
        
        for idx in sorted_indices:
            user_id_similar = self.rating_matrix.index[idx]
            similarity = similarities[idx]
            similar_users.append((user_id_similar, similarity))
        
        return similar_users
    
    def get_similar_items(self, item_id, n_similar=5):
        """Get most similar items"""
        if item_id not in self.rating_matrix.columns:
            return []
        
        item_idx = self.rating_matrix.columns.get_loc(item_id)
        similarities = self.item_similarity[item_idx]
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1][1:n_similar+1]  # Exclude self
        similar_items = []
        
        for idx in sorted_indices:
            item_id_similar = self.rating_matrix.columns[idx]
            similarity = similarities[idx]
            similar_items.append((item_id_similar, similarity))
        
        return similar_items

# Generate synthetic data
np.random.seed(42)
n_users = 100
n_items = 50
n_ratings = 1000

# Create synthetic ratings with some structure
ratings_data = []
for user_id in range(n_users):
    n_user_ratings = np.random.randint(5, 20)
    rated_items = np.random.choice(n_items, n_user_ratings, replace=False)
    
    for item_id in rated_items:
        # Simulate user preferences (some users prefer certain item ranges)
        if user_id < 30:  # First group prefers items 0-15
            base_rating = 4 if item_id < 15 else 2
        elif user_id < 60:  # Second group prefers items 15-30
            base_rating = 4 if 15 <= item_id < 30 else 2
        else:  # Third group prefers items 30+
            base_rating = 4 if item_id >= 30 else 2
        
        # Add noise
        rating = max(1, min(5, base_rating + np.random.normal(0, 0.5)))
        ratings_data.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating
        })

ratings_df = pd.DataFrame(ratings_data)

print("Synthetic Ratings Dataset:")
print(f"Number of users: {n_users}")
print(f"Number of items: {n_items}")
print(f"Number of ratings: {len(ratings_df)}")
print(f"Sparsity: {1 - len(ratings_df) / (n_users * n_items):.3f}")

# Test different collaborative filtering approaches
methods = ['user', 'item']
similarity_metrics = ['cosine', 'pearson']
results = {}

for method in methods:
    for metric in similarity_metrics:
        print(f"\n=== Testing {method.upper()}-based CF with {metric.upper()} similarity ===")
        
        # Initialize and fit model
        cf_model = CollaborativeFiltering(method=method, similarity_metric=metric, k_neighbors=10)
        cf_model.fit(ratings_df)
        
        # Test predictions for a sample user
        test_user = 0
        recommendations = cf_model.recommend(test_user, n_recommendations=5)
        
        print(f"Top 5 recommendations for User {test_user}:")
        for i, (item_id, pred_rating) in enumerate(recommendations):
            print(f"  {i+1}. Item {item_id}: Predicted rating = {pred_rating:.3f}")
        
        # Get similar users/items
        if method == 'user':
            similar_entities = cf_model.get_similar_users(test_user, n_similar=3)
            print(f"Most similar users to User {test_user}:")
        else:
            test_item = 0
            similar_entities = cf_model.get_similar_items(test_item, n_similar=3)
            print(f"Most similar items to Item {test_item}:")
        
        for entity_id, similarity in similar_entities:
            print(f"  {entity_id}: Similarity = {similarity:.3f}")
        
        # Store results for comparison
        results[f"{method}_{metric}"] = {
            'recommendations': recommendations,
            'similar_entities': similar_entities
        }

# Visualization
plt.figure(figsize=(15, 10))

# Plot 1: Rating matrix heatmap (sample)
plt.subplot(2, 3, 1)
sample_matrix = ratings_df.pivot_table(
    index='user_id', columns='item_id', values='rating', fill_value=np.nan
).iloc[:20, :20]
sns.heatmap(sample_matrix, cmap='viridis', cbar_kws={'label': 'Rating'})
plt.title('Rating Matrix (Sample)')
plt.xlabel('Item ID')
plt.ylabel('User ID')

# Plot 2: User similarity matrix (sample)
plt.subplot(2, 3, 2)
user_cf = CollaborativeFiltering(method='user', similarity_metric='cosine')
user_cf.fit(ratings_df)
sample_user_sim = user_cf.user_similarity[:20, :20]
sns.heatmap(sample_user_sim, cmap='coolwarm', center=0, cbar_kws={'label': 'Similarity'})
plt.title('User Similarity Matrix (Sample)')
plt.xlabel('User ID')
plt.ylabel('User ID')

# Plot 3: Item similarity matrix (sample)
plt.subplot(2, 3, 3)
item_cf = CollaborativeFiltering(method='item', similarity_metric='cosine')
item_cf.fit(ratings_df)
sample_item_sim = item_cf.item_similarity[:20, :20]
sns.heatmap(sample_item_sim, cmap='coolwarm', center=0, cbar_kws={'label': 'Similarity'})
plt.title('Item Similarity Matrix (Sample)')
plt.xlabel('Item ID')
plt.ylabel('Item ID')

# Plot 4: Rating distribution
plt.subplot(2, 3, 4)
ratings_df['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')

# Plot 5: Method comparison - predicted ratings
plt.subplot(2, 3, 5)
methods_comparison = []
for key, result in results.items():
    method, metric = key.split('_')
    pred_ratings = [rating for _, rating in result['recommendations']]
    methods_comparison.append({
        'method': f"{method.upper()}-{metric.upper()}",
        'ratings': pred_ratings
    })

for i, method_data in enumerate(methods_comparison):
    plt.boxplot(method_data['ratings'], positions=[i], labels=[method_data['method']])

plt.title('Predicted Ratings by Method')
plt.ylabel('Predicted Rating')
plt.xticks(rotation=45)

# Plot 6: Similarity distribution
plt.subplot(2, 3, 6)
similarities = []
for key, result in results.items():
    method, metric = key.split('_')
    entity_similarities = [sim for _, sim in result['similar_entities']]
    similarities.extend(entity_similarities)

plt.hist(similarities, bins=20, alpha=0.7, edgecolor='black')
plt.title('Similarity Distribution')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Performance comparison
print("\n=== Performance Comparison ===")
for key, result in results.items():
    method, metric = key.split('_')
    pred_ratings = [rating for _, rating in result['recommendations']]
    print(f"{method.upper()}-{metric.upper()}:")
    print(f"  Mean predicted rating: {np.mean(pred_ratings):.3f}")
    print(f"  Std predicted rating: {np.std(pred_ratings):.3f}")
    print(f"  Max predicted rating: {np.max(pred_ratings):.3f}")
    print(f"  Min predicted rating: {np.min(pred_ratings):.3f}")
    print()