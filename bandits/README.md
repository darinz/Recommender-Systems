# Movie Recommendation System using Multi-Armed Bandits

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)](https://github.com/your-repo)
[![Dependencies](https://img.shields.io/badge/Dependencies-NumPy%2C%20Pandas%2C%20Scikit--learn-lightgrey.svg)](https://pypi.org/)

A sophisticated movie recommendation system that leverages multi-armed bandit algorithms to balance exploration of new movies with exploitation of known good recommendations. This system addresses key challenges in recommendation systems including cold-start problems, personalization, and real-time learning.

## Key Features

- **Multi-Armed Bandit Algorithms**: Implements epsilon-greedy, UCB (Upper Confidence Bound), and Thompson sampling
- **Hybrid Recommendation Engine**: Combines collaborative filtering and content-based filtering
- **Cold-Start Handling**: Robust handling of new users and movies
- **Real-time Learning**: Continuously updates recommendations based on user feedback
- **Personalization**: Adapts to individual user preferences over time
- **Performance Evaluation**: Comprehensive metrics and visualization tools

## Algorithms Implemented

### 1. Epsilon-Greedy
- Balances exploration (ε) and exploitation (1-ε)
- Random exploration with greedy exploitation
- Simple and effective for many scenarios

### 2. Upper Confidence Bound (UCB)
- Uses uncertainty estimates to guide exploration
- Automatically balances exploration vs exploitation
- Theoretical guarantees on regret bounds

### 3. Thompson Sampling
- Bayesian approach using posterior sampling
- Naturally handles uncertainty in recommendations
- Often outperforms other bandit algorithms

### 4. Hybrid Approach
- Combines multiple recommendation signals
- Adaptive exploration strategies
- Best performance in practice

## Recommendation Methods

### Collaborative Filtering
- User-based similarity using Pearson correlation
- Finds similar users and leverages their preferences
- Handles sparse data with common movie overlap

### Content-Based Filtering
- Movie feature extraction (genres, year, rating, popularity)
- User preference modeling from rating history
- Cosine similarity between user and movie features

### Hybrid Scoring
- Weighted combination of collaborative and content-based scores
- Adaptive weights based on data availability
- Robust performance across different scenarios

## Quick Start

```python
from recommendation_system import MovieRecommendationBandit

# Initialize the system
recommender = MovieRecommendationBandit(
    num_movies=500,
    num_users=200,
    feature_dim=50,
    exploration_rate=0.1
)

# Get a recommendation for a user
movie_id = recommender.epsilon_greedy_recommend(user_id=1)

# Receive user feedback
recommender.receive_rating(user_id=1, movie_id=movie_id, rating=4.5)

# Run a complete recommendation session
session = recommender.simulate_recommendation_session(
    user_id=1, 
    num_recommendations=20, 
    algorithm='hybrid'
)
```

## Evaluation Metrics

The system evaluates performance using multiple metrics:

- **Average Rating**: Mean rating per recommendation
- **Cumulative Rating**: Total rating across all recommendations
- **Exploration Rate**: Percentage of exploratory recommendations
- **Coverage**: Diversity of recommended movies
- **Regret**: Difference from optimal recommendations

## Cold-Start Handling

### New User Cold-Start
- Uses content-based features when no rating history exists
- Gradually builds preference profile through interactions
- Leverages movie metadata for initial recommendations

### New Movie Cold-Start
- Relies on content-based similarity to existing movies
- Explored through bandit algorithms for discovery
- Builds reputation through user feedback

## Personalization Features

The system learns user preferences through:

- **Genre Preferences**: Analyzes rating patterns by genre
- **Rating Patterns**: Learns user's rating scale and preferences
- **Temporal Preferences**: Adapts to changing preferences over time
- **Feature Learning**: Extracts latent user preferences from interactions

## Usage Examples

### Basic Recommendation
```python
# Get a recommendation using epsilon-greedy
movie_id = recommender.epsilon_greedy_recommend(user_id=1, epsilon=0.2)

# Get a recommendation using UCB
movie_id = recommender.ucb_recommend(user_id=1, alpha=2.0)

# Get a recommendation using Thompson sampling
movie_id = recommender.thompson_sampling_recommend(user_id=1)
```

### Complete Evaluation
```python
# Evaluate all algorithms
results = recommender.evaluate_recommendations(
    num_users=50,
    recommendations_per_user=20
)

# Visualize results
recommender.plot_evaluation_results(results)
```

### Cold-Start Demonstration
```python
# Demonstrate cold-start handling
recommender.demonstrate_cold_start()

# Demonstrate personalization
recommender.demonstrate_personalization()
```

## Configuration Options

### System Parameters
- `num_movies`: Number of movies in the system
- `num_users`: Number of users in the system
- `feature_dim`: Dimension of feature vectors
- `exploration_rate`: Initial exploration rate for epsilon-greedy

### Algorithm Parameters
- `epsilon`: Exploration rate for epsilon-greedy (0.0-1.0)
- `alpha`: Exploration parameter for UCB (typically 1.0-3.0)
- `cf_weight`: Weight for collaborative filtering in hybrid (0.0-1.0)
- `cb_weight`: Weight for content-based filtering in hybrid (0.0-1.0)

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Key Insights

1. **Hybrid algorithms** often perform best by combining multiple signals
2. **Exploration is crucial** for discovering new good movies
3. **Personalization improves** with more user interaction data
4. **Cold-start problems** can be mitigated with content-based features
5. **Bandit algorithms** naturally balance exploration and exploitation

## Advanced Features

### Feature Engineering
- Genre one-hot encoding
- Year normalization
- Rating and popularity features
- Random features for diversity

### User Modeling
- Preference vector learning
- Genre preference analysis
- Rating pattern recognition
- Temporal preference tracking

### Movie Modeling
- Content-based feature extraction
- Genre-based categorization
- Quality and popularity metrics
- Metadata integration

## Performance Characteristics

- **Scalability**: Handles thousands of movies and users
- **Real-time**: Updates recommendations instantly
- **Memory Efficient**: Uses sparse representations
- **Robust**: Handles missing data gracefully

## Contributing

This system is designed for research and educational purposes. Key areas for improvement:

- Integration with real movie datasets
- Advanced feature engineering
- A/B testing framework
- Production deployment considerations
- Additional bandit algorithms

## References

- Multi-armed bandit algorithms for recommendation systems
- Collaborative filtering techniques
- Content-based recommendation methods
- Cold-start problem solutions
- Exploration vs exploitation trade-offs

---

*This system demonstrates how multi-armed bandits can effectively solve the exploration-exploitation dilemma in recommendation systems, providing both immediate value and long-term learning capabilities.* 