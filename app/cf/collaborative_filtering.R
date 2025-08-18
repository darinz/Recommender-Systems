# Collaborative Filtering in R
library(recommenderlab)
library(ggplot2)
library(dplyr)
library(tidyr)

# Generate synthetic data
set.seed(42)
n_users <- 100
n_items <- 50
n_ratings <- 1000

# Create synthetic ratings with structure
ratings_data <- list()
for (user_id in 1:n_users) {
  n_user_ratings <- sample(5:20, 1)
  rated_items <- sample(1:n_items, n_user_ratings, replace = FALSE)
  
  for (item_id in rated_items) {
    # Simulate user preferences
    if (user_id <= 30) {
      base_rating <- ifelse(item_id <= 15, 4, 2)
    } else if (user_id <= 60) {
      base_rating <- ifelse(item_id > 15 && item_id <= 30, 4, 2)
    } else {
      base_rating <- ifelse(item_id > 30, 4, 2)
    }
    
    # Add noise
    rating <- max(1, min(5, base_rating + rnorm(1, 0, 0.5)))
    
    ratings_data[[length(ratings_data) + 1]] <- list(
      user_id = user_id,
      item_id = item_id,
      rating = rating
    )
  }
}

ratings_df <- do.call(rbind, lapply(ratings_data, as.data.frame))

# Create rating matrix
rating_matrix <- ratings_df %>%
  spread(item_id, rating, fill = NA) %>%
  select(-user_id) %>%
  as.matrix()

# Convert to realRatingMatrix
rating_matrix_real <- as(rating_matrix, "realRatingMatrix")

# Test different collaborative filtering methods
methods <- c("UBCF", "IBCF")

results <- list()

for (method in methods) {
  cat("=== Testing", method, "===\n")
  
  # Train model
  model <- Recommender(rating_matrix_real, method = method)
  
  # Generate recommendations
  recommendations <- predict(model, rating_matrix_real[1:5], n = 5)
  
  # Display recommendations
  for (i in 1:5) {
    cat("User", i, "recommendations:", as(recommendations[i], "list")[[1]], "\n")
  }
  
  # Store results
  results[[method]] <- model
}

# Visualization
# Rating distribution
p1 <- ggplot(ratings_df, aes(x = factor(rating))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Rating Distribution",
       x = "Rating", y = "Count") +
  theme_minimal()

# User-item matrix heatmap (sample)
sample_matrix <- rating_matrix[1:20, 1:20]
sample_df <- expand.grid(
  user_id = 1:20,
  item_id = 1:20
)
sample_df$rating <- as.vector(sample_matrix)

p2 <- ggplot(sample_df, aes(x = item_id, y = user_id, fill = rating)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(title = "Rating Matrix (Sample)",
       x = "Item ID", y = "User ID") +
  theme_minimal()

# Combine plots
library(gridExtra)
grid.arrange(p1, p2, ncol = 2)