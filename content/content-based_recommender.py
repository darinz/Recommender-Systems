import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    def __init__(self, similarity_metric='cosine'):
        """
        Content-Based Recommender System
        
        Parameters:
        -----------
        similarity_metric : str
            Similarity metric ('cosine', 'euclidean', 'pearson')
        """
        self.similarity_metric = similarity_metric
        self.item_profiles = None
        self.user_profiles = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def _compute_similarity(self, profile1, profile2):
        """Compute similarity between two profiles"""
        if self.similarity_metric == 'cosine':
            return cosine_similarity([profile1], [profile2])[0][0]
        elif self.similarity_metric == 'euclidean':
            distance = np.linalg.norm(profile1 - profile2)
            return 1 / (1 + distance)
        elif self.similarity_metric == 'pearson':
            return np.corrcoef(profile1, profile2)[0, 1]
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def create_item_profiles(self, items_df, feature_columns, text_columns=None):
        """
        Create item profiles from item features
        
        Parameters:
        -----------
        items_df : pandas.DataFrame
            DataFrame containing item features
        feature_columns : list
            List of feature column names
        text_columns : list, optional
            List of text column names for TF-IDF
        """
        profiles = []
        feature_names = []
        
        # Handle categorical features
        for col in feature_columns:
            if items_df[col].dtype == 'object':
                # Encode categorical features
                le = LabelEncoder()
                encoded_values = le.fit_transform(items_df[col])
                profiles.append(encoded_values)
                feature_names.extend([f"{col}_{val}" for val in le.classes_])
                self.label_encoders[col] = le
            else:
                # Numerical features
                profiles.append(items_df[col].values)
                feature_names.append(col)
        
        # Handle text features
        if text_columns:
            for col in text_columns:
                tfidf = TfidfVectorizer(max_features=50, stop_words='english')
                text_features = tfidf.fit_transform(items_df[col].fillna(''))
                profiles.append(text_features.toarray())
                feature_names.extend([f"{col}_{word}" for word in tfidf.get_feature_names_out()])
        
        # Combine all features
        self.item_profiles = np.hstack(profiles)
        self.feature_names = feature_names
        
        # Normalize features
        self.item_profiles = self.scaler.fit_transform(self.item_profiles)
        
        return self.item_profiles
    
    def create_user_profiles(self, ratings_df, items_df, user_id_col='user_id', 
                           item_id_col='item_id', rating_col='rating'):
        """
        Create user profiles from ratings and item features
        
        Parameters:
        -----------
        ratings_df : pandas.DataFrame
            DataFrame containing user ratings
        items_df : pandas.DataFrame
            DataFrame containing item features
        """
        if self.item_profiles is None:
            raise ValueError("Item profiles must be created first")
        
        user_profiles = {}
        
        for user_id in ratings_df[user_id_col].unique():
            user_ratings = ratings_df[ratings_df[user_id_col] == user_id]
            
            # Get items rated by this user
            rated_items = user_ratings[item_id_col].values
            ratings = user_ratings[rating_col].values
            
            # Find corresponding item profiles
            item_indices = [items_df.index.get_loc(item_id) for item_id in rated_items]
            item_profiles = self.item_profiles[item_indices]
            
            # Compute weighted average (weighted by ratings)
            weights = ratings / ratings.sum()
            user_profile = np.average(item_profiles, weights=weights, axis=0)
            
            user_profiles[user_id] = user_profile
        
        self.user_profiles = user_profiles
        return user_profiles
    
    def recommend(self, user_id, n_recommendations=5, exclude_rated=True):
        """
        Generate recommendations for a user
        
        Parameters:
        -----------
        user_id : int
            User ID to generate recommendations for
        n_recommendations : int
            Number of recommendations to generate
        exclude_rated : bool
            Whether to exclude items the user has already rated
        """
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # Compute similarities with all items
        similarities = []
        for i, item_profile in enumerate(self.item_profiles):
            similarity = self._compute_similarity(user_profile, item_profile)
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top recommendations
        return similarities[:n_recommendations]
    
    def get_feature_importance(self, user_id, top_features=10):
        """Get most important features for a user"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # Get feature importance (absolute values)
        feature_importance = [(name, abs(value)) for name, value in zip(self.feature_names, user_profile)]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:top_features]
    
    def visualize_profiles(self, user_ids=None, n_items=20):
        """Visualize user and item profiles using PCA"""
        if user_ids is None:
            user_ids = list(self.user_profiles.keys())[:5]
        
        # Combine user and item profiles
        all_profiles = []
        profile_labels = []
        profile_types = []
        
        # Add user profiles
        for user_id in user_ids:
            all_profiles.append(self.user_profiles[user_id])
            profile_labels.append(f"User {user_id}")
            profile_types.append("User")
        
        # Add item profiles (sample)
        item_indices = np.random.choice(len(self.item_profiles), n_items, replace=False)
        for idx in item_indices:
            all_profiles.append(self.item_profiles[idx])
            profile_labels.append(f"Item {idx}")
            profile_types.append("Item")
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        profiles_2d = pca.fit_transform(all_profiles)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot users and items
        for i, (profile, label, profile_type) in enumerate(zip(profiles_2d, profile_labels, profile_types)):
            if profile_type == "User":
                plt.scatter(profile[0], profile[1], c='red', s=100, marker='s', label=label if i < len(user_ids) else "")
            else:
                plt.scatter(profile[0], profile[1], c='blue', s=50, alpha=0.6)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('User and Item Profiles in 2D Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Generate synthetic movie data
np.random.seed(42)
n_movies = 100
n_users = 50

# Create movie features
movies_df = pd.DataFrame({
    'movie_id': range(n_movies),
    'title': [f'Movie_{i}' for i in range(n_movies)],
    'genre': np.random.choice(['Action', 'Drama', 'Comedy', 'Thriller', 'Romance'], n_movies),
    'year': np.random.randint(1990, 2024, n_movies),
    'rating': np.random.uniform(1, 10, n_movies),
    'budget': np.random.uniform(1, 100, n_movies),
    'director': np.random.choice(['Spielberg', 'Nolan', 'Tarantino', 'Scorsese', 'Cameron'], n_movies),
    'description': [f'Description for movie {i}' for i in range(n_movies)]
})

# Create synthetic ratings
ratings_data = []
for user_id in range(n_users):
    n_ratings = np.random.randint(5, 20)
    rated_movies = np.random.choice(n_movies, n_ratings, replace=False)
    
    for movie_id in rated_movies:
        # Simulate user preferences based on movie features
        movie = movies_df.iloc[movie_id]
        base_rating = 5
        
        # Genre preferences (simulate user taste)
        if movie['genre'] in ['Action', 'Thriller']:
            base_rating += np.random.normal(1, 1)
        elif movie['genre'] in ['Drama', 'Romance']:
            base_rating += np.random.normal(-1, 1)
        
        # Year preference (prefer newer movies)
        year_factor = (movie['year'] - 1990) / (2024 - 1990)
        base_rating += year_factor * 2
        
        # Add noise
        rating = max(1, min(10, base_rating + np.random.normal(0, 1)))
        ratings_data.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating
        })

ratings_df = pd.DataFrame(ratings_data)

print("Synthetic Movie Dataset:")
print(f"Number of movies: {n_movies}")
print(f"Number of users: {n_users}")
print(f"Number of ratings: {len(ratings_df)}")

# Initialize and train content-based recommender
recommender = ContentBasedRecommender(similarity_metric='cosine')

# Create item profiles
feature_columns = ['genre', 'year', 'rating', 'budget', 'director']
text_columns = ['description']
item_profiles = recommender.create_item_profiles(movies_df, feature_columns, text_columns)

print(f"\nItem profiles shape: {item_profiles.shape}")
print(f"Number of features: {len(recommender.feature_names)}")

# Create user profiles
user_profiles = recommender.create_user_profiles(ratings_df, movies_df)

print(f"Number of user profiles: {len(user_profiles)}")

# Generate recommendations for a sample user
test_user = 0
recommendations = recommender.recommend(test_user, n_recommendations=10)

print(f"\nTop 10 recommendations for User {test_user}:")
for i, (item_idx, similarity) in enumerate(recommendations):
    movie = movies_df.iloc[item_idx]
    print(f"{i+1}. {movie['title']} ({movie['genre']}, {movie['year']}) - Similarity: {similarity:.3f}")

# Get feature importance for the user
feature_importance = recommender.get_feature_importance(test_user, top_features=10)

print(f"\nTop 10 most important features for User {test_user}:")
for feature, importance in feature_importance:
    print(f"  {feature}: {importance:.3f}")

# Visualize profiles
recommender.visualize_profiles(user_ids=[0, 1, 2], n_items=30)

# Compare different similarity metrics
similarity_metrics = ['cosine', 'euclidean', 'pearson']
results = {}

for metric in similarity_metrics:
    print(f"\n=== Testing {metric.upper()} Similarity ===")
    
    recommender_metric = ContentBasedRecommender(similarity_metric=metric)
    recommender_metric.create_item_profiles(movies_df, feature_columns, text_columns)
    recommender_metric.create_user_profiles(ratings_df, movies_df)
    
    recommendations = recommender_metric.recommend(test_user, n_recommendations=5)
    results[metric] = recommendations
    
    print(f"Top 5 recommendations:")
    for i, (item_idx, similarity) in enumerate(recommendations):
        movie = movies_df.iloc[item_idx]
        print(f"  {i+1}. {movie['title']} - Similarity: {similarity:.3f}")

# Visualization of similarity distributions
plt.figure(figsize=(15, 5))

for i, (metric, recommendations) in enumerate(results.items()):
    similarities = [sim for _, sim in recommendations]
    
    plt.subplot(1, 3, i+1)
    plt.hist(similarities, bins=10, alpha=0.7, edgecolor='black')
    plt.title(f'{metric.capitalize()} Similarity Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()