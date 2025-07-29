# Challenges in Recommender Systems - R Implementation
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Generate synthetic data with challenges
set.seed(42)
n_users <- 1000
n_items <- 500
n_ratings <- 5000

# Create synthetic ratings with challenges
ratings_data <- list()

# Create popular items and active users
popular_items <- sample(1:n_items, 50, replace = FALSE)
active_users <- sample(1:n_users, 100, replace = FALSE)

for (user_id in 1:n_users) {
  # Vary number of ratings based on user activity
  if (user_id %in% active_users) {
    n_user_ratings <- sample(20:50, 1)
  } else {
    n_user_ratings <- sample(1:10, 1)
  }
  
  rated_items <- sample(1:n_items, n_user_ratings, replace = FALSE)
  
  for (item_id in rated_items) {
    # Create popularity bias
    if (item_id %in% popular_items) {
      base_rating <- rnorm(1, 4.0, 0.5)
    } else {
      base_rating <- rnorm(1, 3.0, 0.8)
    }
    
    # Add cold start users
    if (runif(1) < 0.1) {
      base_rating <- rnorm(1, 3.0, 1.0)
    }
    
    rating <- max(1, min(5, base_rating))
    
    ratings_data[[length(ratings_data) + 1]] <- list(
      user_id = user_id,
      item_id = item_id,
      rating = rating
    )
  }
}

ratings_df <- do.call(rbind, lapply(ratings_data, as.data.frame))

# Analyze challenges
# Cold start analysis
user_counts <- ratings_df %>%
  group_by(user_id) %>%
  summarise(n_ratings = n()) %>%
  arrange(desc(n_ratings))

item_counts <- ratings_df %>%
  group_by(item_id) %>%
  summarise(n_ratings = n()) %>%
  arrange(desc(n_ratings))

cold_start_users <- sum(user_counts$n_ratings <= 1)
cold_start_items <- sum(item_counts$n_ratings <= 1)

# Sparsity analysis
total_entries <- n_users * n_items
observed_entries <- nrow(ratings_df)
sparsity <- 1 - (observed_entries / total_entries)

# Popularity bias analysis
item_popularity <- ratings_df %>%
  group_by(item_id) %>%
  summarise(n_ratings = n()) %>%
  arrange(desc(n_ratings))

# Calculate Gini coefficient
calculate_gini <- function(values) {
  sorted_values <- sort(values)
  n <- length(sorted_values)
  cumsum_values <- cumsum(sorted_values)
  return((n + 1 - 2 * sum(cumsum_values) / cumsum_values[n]) / n)
}

gini_coefficient <- calculate_gini(item_popularity$n_ratings)

# Visualization
# Cold start analysis
p1 <- ggplot(user_counts, aes(x = n_ratings)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
  labs(title = "User Rating Distribution",
       x = "Number of Ratings", y = "Frequency") +
  theme_minimal()

p2 <- ggplot(item_counts, aes(x = n_ratings)) +
  geom_histogram(bins = 30, fill = "lightcoral", alpha = 0.7) +
  labs(title = "Item Rating Distribution",
       x = "Number of Ratings", y = "Frequency") +
  theme_minimal()

# Popularity bias
p3 <- ggplot(head(item_popularity, 20), aes(x = reorder(factor(item_id), n_ratings), y = n_ratings)) +
  geom_bar(stat = "identity", fill = "orange") +
  labs(title = "Top 20 Most Popular Items",
       x = "Item ID", y = "Number of Ratings") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Rating distribution
p4 <- ggplot(ratings_df, aes(x = factor(rating))) +
  geom_bar(fill = "green", alpha = 0.7) +
  labs(title = "Rating Distribution",
       x = "Rating", y = "Count") +
  theme_minimal()

# Combine plots
grid.arrange(p1, p2, p3, p4, ncol = 2)

# Print analysis results
cat("=== Challenge Analysis Results ===\n")
cat("Cold Start Analysis:\n")
cat("Cold start users:", cold_start_users, "\n")
cat("Cold start items:", cold_start_items, "\n")
cat("User cold start rate:", cold_start_users / n_users, "\n")
cat("Item cold start rate:", cold_start_items / n_items, "\n")

cat("\nSparsity Analysis:\n")
cat("Sparsity:", sparsity, "\n")
cat("Total entries:", total_entries, "\n")
cat("Observed entries:", observed_entries, "\n")

cat("\nPopularity Bias Analysis:\n")
cat("Gini coefficient:", gini_coefficient, "\n")
cat("Top 10% items share:", sum(head(item_popularity$n_ratings, n_items * 0.1)) / observed_entries, "\n")
cat("Bottom 10% items share:", sum(tail(item_popularity$n_ratings, n_items * 0.1)) / observed_entries, "\n")