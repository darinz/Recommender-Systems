# Challenge Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class RecommenderSystemChallenges:
    """Analysis of common challenges in recommender systems"""
    
    def __init__(self):
        self.challenges = {}
        
    def analyze_cold_start(self, ratings_df, user_col='user_id', item_col='item_id'):
        """Analyze cold start problem"""
        # Count ratings per user and item
        user_rating_counts = ratings_df[user_col].value_counts()
        item_rating_counts = ratings_df[item_col].value_counts()
        
        # Identify cold start cases
        cold_start_users = user_rating_counts[user_rating_counts <= 1]
        cold_start_items = item_rating_counts[item_rating_counts <= 1]
        
        # Calculate statistics
        total_users = len(user_rating_counts)
        total_items = len(item_rating_counts)
        
        cold_start_stats = {
            'cold_start_users': len(cold_start_users),
            'cold_start_items': len(cold_start_items),
            'user_cold_start_rate': len(cold_start_users) / total_users,
            'item_cold_start_rate': len(cold_start_items) / total_items,
            'avg_ratings_per_user': user_rating_counts.mean(),
            'avg_ratings_per_item': item_rating_counts.mean(),
            'median_ratings_per_user': user_rating_counts.median(),
            'median_ratings_per_item': item_rating_counts.median()
        }
        
        return cold_start_stats, user_rating_counts, item_rating_counts
    
    def analyze_sparsity(self, ratings_df, user_col='user_id', item_col='item_id'):
        """Analyze data sparsity"""
        # Create rating matrix
        rating_matrix = ratings_df.pivot_table(
            index=user_col, 
            columns=item_col, 
            values='rating', 
            fill_value=np.nan
        )
        
        # Calculate sparsity
        total_entries = rating_matrix.shape[0] * rating_matrix.shape[1]
        observed_entries = (~rating_matrix.isna()).sum().sum()
        sparsity = 1 - (observed_entries / total_entries)
        
        # Analyze rating distribution
        rating_distribution = ratings_df['rating'].value_counts().sort_index()
        
        # Calculate coverage metrics
        user_coverage = (~rating_matrix.isna()).sum(axis=1)
        item_coverage = (~rating_matrix.isna()).sum(axis=0)
        
        sparsity_stats = {
            'sparsity': sparsity,
            'total_entries': total_entries,
            'observed_entries': observed_entries,
            'avg_user_coverage': user_coverage.mean(),
            'avg_item_coverage': item_coverage.mean(),
            'min_user_coverage': user_coverage.min(),
            'max_user_coverage': user_coverage.max(),
            'min_item_coverage': item_coverage.min(),
            'max_item_coverage': item_coverage.max()
        }
        
        return sparsity_stats, rating_matrix, rating_distribution
    
    def analyze_popularity_bias(self, ratings_df, user_col='user_id', item_col='item_id'):
        """Analyze popularity bias"""
        # Calculate item popularity
        item_popularity = ratings_df[item_col].value_counts()
        
        # Calculate user activity
        user_activity = ratings_df[user_col].value_counts()
        
        # Calculate popularity bias metrics
        gini_coefficient_items = self._calculate_gini(item_popularity.values)
        gini_coefficient_users = self._calculate_gini(user_activity.values)
        
        # Calculate recommendation diversity
        top_items = item_popularity.head(10)
        bottom_items = item_popularity.tail(10)
        
        popularity_stats = {
            'gini_coefficient_items': gini_coefficient_items,
            'gini_coefficient_users': gini_coefficient_users,
            'top_10_items_share': top_items.sum() / item_popularity.sum(),
            'bottom_10_items_share': bottom_items.sum() / item_popularity.sum(),
            'popularity_ratio': item_popularity.max() / item_popularity.min(),
            'activity_ratio': user_activity.max() / user_activity.min()
        }
        
        return popularity_stats, item_popularity, user_activity
    
    def _calculate_gini(self, values):
        """Calculate Gini coefficient"""
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def analyze_scalability(self, ratings_df, user_col='user_id', item_col='item_id'):
        """Analyze scalability challenges"""
        n_users = ratings_df[user_col].nunique()
        n_items = ratings_df[item_col].nunique()
        n_ratings = len(ratings_df)
        
        # Calculate computational complexity estimates
        ubcf_complexity = n_users ** 2 * n_items
        ibcf_complexity = n_items ** 2 * n_users
        mf_complexity = n_ratings * 10 * 100  # Assuming 10 factors, 100 epochs
        
        # Memory requirements
        user_sim_memory = n_users ** 2 * 8  # 8 bytes per float
        item_sim_memory = n_items ** 2 * 8
        
        scalability_stats = {
            'n_users': n_users,
            'n_items': n_items,
            'n_ratings': n_ratings,
            'ubcf_complexity': ubcf_complexity,
            'ibcf_complexity': ibcf_complexity,
            'mf_complexity': mf_complexity,
            'user_sim_memory_mb': user_sim_memory / (1024 * 1024),
            'item_sim_memory_mb': item_sim_memory / (1024 * 1024),
            'user_item_ratio': n_users / n_items,
            'density': n_ratings / (n_users * n_items)
        }
        
        return scalability_stats
    
    def simulate_cold_start_impact(self, ratings_df, user_col='user_id', item_col='item_id', 
                                 rating_col='rating', test_fraction=0.1):
        """Simulate impact of cold start on recommendation quality"""
        from sklearn.model_selection import train_test_split
        
        # Split data
        train_df, test_df = train_test_split(ratings_df, test_size=test_fraction, random_state=42)
        
        # Identify cold start cases in test set
        train_users = set(train_df[user_col].unique())
        train_items = set(train_df[item_col].unique())
        
        cold_start_test = test_df[
            (~test_df[user_col].isin(train_users)) | 
            (~test_df[item_col].isin(train_items))
        ]
        
        regular_test = test_df[
            (test_df[user_col].isin(train_users)) & 
            (test_df[item_col].isin(train_items))
        ]
        
        # Calculate baseline predictions
        global_mean = train_df[rating_col].mean()
        
        # Evaluate on different test sets
        cold_start_mae = mean_absolute_error(
            cold_start_test[rating_col], 
            [global_mean] * len(cold_start_test)
        )
        
        regular_mae = mean_absolute_error(
            regular_test[rating_col], 
            [global_mean] * len(regular_test)
        )
        
        impact_stats = {
            'cold_start_mae': cold_start_mae,
            'regular_mae': regular_mae,
            'cold_start_ratio': len(cold_start_test) / len(test_df),
            'performance_degradation': cold_start_mae / regular_mae if regular_mae > 0 else float('inf')
        }
        
        return impact_stats, cold_start_test, regular_test
    
    def analyze_bias_mitigation(self, ratings_df, user_col='user_id', item_col='item_id', 
                              rating_col='rating'):
        """Analyze bias mitigation strategies"""
        # Calculate item popularity
        item_popularity = ratings_df[item_col].value_counts()
        
        # Calculate popularity bias
        popularity_bias = item_popularity / item_popularity.sum()
        
        # Apply debiasing techniques
        # 1. Inverse popularity sampling
        inverse_popularity = 1 / (item_popularity + 1)  # Add 1 to avoid division by zero
        debiased_popularity = inverse_popularity / inverse_popularity.sum()
        
        # 2. Square root debiasing
        sqrt_popularity = np.sqrt(item_popularity)
        sqrt_debiased = sqrt_popularity / sqrt_popularity.sum()
        
        # 3. Log debiasing
        log_popularity = np.log(item_popularity + 1)
        log_debiased = log_popularity / log_popularity.sum()
        
        bias_mitigation_stats = {
            'original_gini': self._calculate_gini(item_popularity.values),
            'inverse_gini': self._calculate_gini(debiased_popularity.values),
            'sqrt_gini': self._calculate_gini(sqrt_debiased.values),
            'log_gini': self._calculate_gini(log_debiased.values),
            'popularity_correlation': np.corrcoef(item_popularity.values, 
                                                range(len(item_popularity)))[0, 1]
        }
        
        return bias_mitigation_stats, {
            'original': popularity_bias,
            'inverse': debiased_popularity,
            'sqrt': sqrt_debiased,
            'log': log_debiased
        }

# Generate synthetic data with various challenges
np.random.seed(42)
n_users = 1000
n_items = 500
n_ratings = 5000

# Create synthetic ratings with challenges
ratings_data = []

# Create some popular items and active users
popular_items = np.random.choice(n_items, 50, replace=False)
active_users = np.random.choice(n_users, 100, replace=False)

for user_id in range(n_users):
    # Vary number of ratings based on user activity
    if user_id in active_users:
        n_user_ratings = np.random.randint(20, 50)
    else:
        n_user_ratings = np.random.randint(1, 10)
    
    rated_items = np.random.choice(n_items, n_user_ratings, replace=False)
    
    for item_id in rated_items:
        # Create popularity bias
        if item_id in popular_items:
            base_rating = np.random.normal(4.0, 0.5)
        else:
            base_rating = np.random.normal(3.0, 0.8)
        
        # Add some cold start users (few ratings)
        if np.random.random() < 0.1:  # 10% cold start users
            base_rating = np.random.normal(3.0, 1.0)
        
        rating = max(1, min(5, base_rating))
        ratings_data.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating
        })

ratings_df = pd.DataFrame(ratings_data)

print("Synthetic Dataset with Challenges:")
print(f"Number of users: {n_users}")
print(f"Number of items: {n_items}")
print(f"Number of ratings: {len(ratings_df)}")

# Analyze challenges
challenge_analyzer = RecommenderSystemChallenges()

print("\n=== Cold Start Analysis ===")
cold_start_stats, user_counts, item_counts = challenge_analyzer.analyze_cold_start(ratings_df)
for key, value in cold_start_stats.items():
    print(f"{key}: {value:.4f}")

print("\n=== Sparsity Analysis ===")
sparsity_stats, rating_matrix, rating_dist = challenge_analyzer.analyze_sparsity(ratings_df)
for key, value in sparsity_stats.items():
    print(f"{key}: {value:.4f}")

print("\n=== Popularity Bias Analysis ===")
popularity_stats, item_popularity, user_activity = challenge_analyzer.analyze_popularity_bias(ratings_df)
for key, value in popularity_stats.items():
    print(f"{key}: {value:.4f}")

print("\n=== Scalability Analysis ===")
scalability_stats = challenge_analyzer.analyze_scalability(ratings_df)
for key, value in scalability_stats.items():
    print(f"{key}: {value:.2f}")

print("\n=== Cold Start Impact Simulation ===")
impact_stats, cold_test, regular_test = challenge_analyzer.simulate_cold_start_impact(ratings_df)
for key, value in impact_stats.items():
    print(f"{key}: {value:.4f}")

print("\n=== Bias Mitigation Analysis ===")
bias_stats, debiased_distributions = challenge_analyzer.analyze_bias_mitigation(ratings_df)
for key, value in bias_stats.items():
    print(f"{key}: {value:.4f}")

# Visualization
plt.figure(figsize=(20, 15))

# Plot 1: Cold start analysis
plt.subplot(3, 4, 1)
plt.hist(user_counts.values, bins=30, alpha=0.7, edgecolor='black')
plt.title('User Rating Distribution')
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')

plt.subplot(3, 4, 2)
plt.hist(item_counts.values, bins=30, alpha=0.7, edgecolor='black')
plt.title('Item Rating Distribution')
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')

# Plot 3: Sparsity visualization
plt.subplot(3, 4, 3)
sample_matrix = rating_matrix.iloc[:50, :50]
sns.heatmap(sample_matrix, cmap='viridis', cbar_kws={'label': 'Rating'})
plt.title('Rating Matrix (Sample)')
plt.xlabel('Item ID')
plt.ylabel('User ID')

# Plot 4: Popularity bias
plt.subplot(3, 4, 4)
top_items = item_popularity.head(20)
plt.bar(range(len(top_items)), top_items.values)
plt.title('Top 20 Most Popular Items')
plt.xlabel('Item Rank')
plt.ylabel('Number of Ratings')

# Plot 5: Scalability analysis
plt.subplot(3, 4, 5)
complexities = ['UBCF', 'IBCF', 'MF']
complexity_values = [
    scalability_stats['ubcf_complexity'] / 1e6,
    scalability_stats['ibcf_complexity'] / 1e6,
    scalability_stats['mf_complexity'] / 1e6
]
plt.bar(complexities, complexity_values)
plt.title('Computational Complexity (Million Operations)')
plt.ylabel('Complexity')

# Plot 6: Memory requirements
plt.subplot(3, 4, 6)
memory_requirements = [
    scalability_stats['user_sim_memory_mb'],
    scalability_stats['item_sim_memory_mb']
]
plt.bar(['User Similarity', 'Item Similarity'], memory_requirements)
plt.title('Memory Requirements (MB)')
plt.ylabel('Memory (MB)')

# Plot 7: Bias mitigation comparison
plt.subplot(3, 4, 7)
gini_values = [
    bias_stats['original_gini'],
    bias_stats['inverse_gini'],
    bias_stats['sqrt_gini'],
    bias_stats['log_gini']
]
methods = ['Original', 'Inverse', 'Sqrt', 'Log']
plt.bar(methods, gini_values)
plt.title('Gini Coefficient by Debiasing Method')
plt.ylabel('Gini Coefficient')

# Plot 8: Cold start impact
plt.subplot(3, 4, 8)
mae_values = [impact_stats['regular_mae'], impact_stats['cold_start_mae']]
plt.bar(['Regular', 'Cold Start'], mae_values)
plt.title('MAE Comparison')
plt.ylabel('Mean Absolute Error')

# Plot 9: Rating distribution
plt.subplot(3, 4, 9)
rating_dist.plot(kind='bar')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')

# Plot 10: User activity distribution
plt.subplot(3, 4, 10)
plt.hist(user_activity.values, bins=30, alpha=0.7, edgecolor='black')
plt.title('User Activity Distribution')
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')

# Plot 11: Sparsity over time simulation
plt.subplot(3, 4, 11)
# Simulate sparsity as system grows
user_sizes = np.arange(100, 1001, 100)
sparsity_values = []
for n in user_sizes:
    sparsity = 1 - (n_ratings / (n * n_items))
    sparsity_values.append(sparsity)

plt.plot(user_sizes, sparsity_values)
plt.title('Sparsity vs System Size')
plt.xlabel('Number of Users')
plt.ylabel('Sparsity')

# Plot 12: Challenge summary
plt.subplot(3, 4, 12)
challenges = ['Cold Start', 'Sparsity', 'Scalability', 'Bias']
severity = [
    cold_start_stats['user_cold_start_rate'],
    sparsity_stats['sparsity'],
    min(1.0, scalability_stats['ubcf_complexity'] / 1e9),  # Normalize
    bias_stats['original_gini']
]
plt.bar(challenges, severity)
plt.title('Challenge Severity')
plt.ylabel('Severity Score')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Detailed analysis
print("\n=== Detailed Challenge Analysis ===")

# Cold start impact by user type
print("Cold Start Impact by User Type:")
active_cold_start = len(set(active_users) & set(cold_test['user_id'].unique()))
inactive_cold_start = len(set(cold_test['user_id'].unique()) - set(active_users))
print(f"Active users in cold start: {active_cold_start}")
print(f"Inactive users in cold start: {inactive_cold_start}")

# Popularity bias analysis
print(f"\nPopularity Bias Analysis:")
print(f"Top 10% items account for {popularity_stats['top_10_items_share']:.2%} of ratings")
print(f"Bottom 10% items account for {popularity_stats['bottom_10_items_share']:.2%} of ratings")
print(f"Popularity ratio: {popularity_stats['popularity_ratio']:.2f}")

# Scalability recommendations
print(f"\nScalability Recommendations:")
if scalability_stats['user_item_ratio'] > 2:
    print("Recommend IBCF (more users than items)")
elif scalability_stats['user_item_ratio'] < 0.5:
    print("Recommend UBCF (more items than users)")
else:
    print("Consider both UBCF and IBCF")

if scalability_stats['user_sim_memory_mb'] > 1000:
    print("User similarity matrix too large - consider sampling")
if scalability_stats['item_sim_memory_mb'] > 1000:
    print("Item similarity matrix too large - consider sampling")

# Bias mitigation effectiveness
print(f"\nBias Mitigation Effectiveness:")
improvements = {
    'Inverse': bias_stats['original_gini'] - bias_stats['inverse_gini'],
    'Sqrt': bias_stats['original_gini'] - bias_stats['sqrt_gini'],
    'Log': bias_stats['original_gini'] - bias_stats['log_gini']
}
best_method = max(improvements, key=improvements.get)
print(f"Best debiasing method: {best_method} (improvement: {improvements[best_method]:.4f})")