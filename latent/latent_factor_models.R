# Latent Factor Models in R
library(recommenderlab)
library(ggplot2)
library(dplyr)
library(tidyr)
library(NMF)

# Generate synthetic data with latent structure
set.seed(42)
n_users <- 300
n_items <- 200
n_ratings <- 3000

# Create synthetic ratings with latent factors
ratings_data <- list()
for (user_id in 1:n_users) {
  n_user_ratings <- sample(8:25, 1)
  rated_items <- sample(1:n_items, n_user_ratings, replace = FALSE)
  
  for (item_id in rated_items) {
    # Create latent factor structure
    user_action_pref <- rnorm(1, 0, 1)
    user_complexity_pref <- rnorm(1, 0, 1)
    user_genre_pref <- rnorm(1, 0, 1)
    
    item_action_level <- rnorm(1, 0, 1)
    item_complexity <- rnorm(1, 0, 1)
    item_genre <- rnorm(1, 0, 1)
    
    # Compute rating based on latent factors
    latent_score <- (user_action_pref * item_action_level + 
                     user_complexity_pref * item_complexity + 
                     user_genre_pref * item_genre)
    
    # Add noise and convert to 1-5 scale
    rating <- max(1, min(5, 3 + latent_score + rnorm(1, 0, 0.5)))
    
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

# Split data for evaluation
set.seed(42)
train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
train_df <- ratings_df[train_indices, ]
test_df <- ratings_df[-train_indices, ]

# Create training matrix
train_matrix <- train_df %>%
  spread(item_id, rating, fill = 0) %>%
  select(-user_id) %>%
  as.matrix()

# Test different latent factor methods
methods <- c("SVD", "NMF")

results <- list()

for (method in methods) {
  cat("Testing", method, "\n")
  
  if (method == "SVD") {
    # SVD-based recommendation
    model <- Recommender(train_matrix_real, method = "SVD")
  } else if (method == "NMF") {
    # Non-negative Matrix Factorization
    nmf_result <- nmf(train_matrix, 10, method = "brunet", nrun = 1)
    # For simplicity, we'll use a basic approach
    model <- list(type = "NMF", factors = nmf_result)
  }
  
  # Generate predictions
  predictions <- predict(model, train_matrix_real[1:min(10, nrow(train_matrix_real))], n = 5)
  
  # Store results
  results[[method]] <- list(
    model = model,
    predictions = predictions
  )
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

# Method comparison
method_names <- names(results)
comparison_df <- data.frame(
  method = method_names,
  mae = c(0.5, 0.6),  # Placeholder values
  rmse = c(0.7, 0.8)  # Placeholder values
)

p3 <- ggplot(comparison_df, aes(x = method, y = mae)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  labs(title = "MAE Comparison",
       x = "Method", y = "Mean Absolute Error") +
  theme_minimal()

p4 <- ggplot(comparison_df, aes(x = method, y = rmse)) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  labs(title = "RMSE Comparison",
       x = "Method", y = "Root Mean Square Error") +
  theme_minimal()

# Combine plots
library(gridExtra)
grid.arrange(p1, p2, p3, p4, ncol = 2)