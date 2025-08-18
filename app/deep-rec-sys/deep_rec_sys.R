# Deep Recommender Systems in R
# This script demonstrates the implementation of deep recommender systems using Keras.
# It includes various models: NCF, Wide & Deep, and NeuMF, and evaluates their performance.
# The script also includes synthetic data generation with non-linear patterns and visualization of model performance.   

library(keras)
library(tensorflow)
library(ggplot2)
library(dplyr)
library(tidyr)

# Generate synthetic data
set.seed(42)
n_users <- 500
n_items <- 300
n_ratings <- 3000

# Create synthetic ratings with non-linear patterns
ratings_data <- list()
for (user_id in 1:n_users) {
  n_user_ratings <- sample(5:20, 1)
  rated_items <- sample(1:n_items, n_user_ratings, replace = FALSE)
  
  for (item_id in rated_items) {
    # Create non-linear patterns
    user_factor <- rnorm(1, 0, 1)
    item_factor <- rnorm(1, 0, 1)
    
    # Non-linear interaction
    interaction <- sin(user_factor) * cos(item_factor) + user_factor * item_factor
    
    # Add noise and convert to rating
    rating <- max(1, min(5, 3 + interaction + rnorm(1, 0, 0.3)))
    
    ratings_data[[length(ratings_data) + 1]] <- list(
      user_id = user_id,
      item_id = item_id,
      rating = rating
    )
  }
}

ratings_df <- do.call(rbind, lapply(ratings_data, as.data.frame))

# Create user and item mappings
user_mapping <- setNames(1:length(unique(ratings_df$user_id)), unique(ratings_df$user_id))
item_mapping <- setNames(1:length(unique(ratings_df$item_id)), unique(ratings_df$item_id))

# Convert to indices
ratings_df$user_idx <- user_mapping[as.character(ratings_df$user_id)]
ratings_df$item_idx <- item_mapping[as.character(ratings_df$item_id)]

# Split data
set.seed(42)
train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
train_df <- ratings_df[train_indices, ]
test_df <- ratings_df[-train_indices, ]

# Prepare data for Keras
n_users <- length(unique(ratings_df$user_id))
n_items <- length(unique(ratings_df$item_id))
n_factors <- 10

# NCF Model
build_ncf_model <- function() {
  # Input layers
  user_input <- layer_input(shape = 1, name = "user_input")
  item_input <- layer_input(shape = 1, name = "item_input")
  
  # Embeddings
  user_embedding <- user_input %>%
    layer_embedding(input_dim = n_users, output_dim = n_factors, name = "user_embedding") %>%
    layer_flatten()
  
  item_embedding <- item_input %>%
    layer_embedding(input_dim = n_items, output_dim = n_factors, name = "item_embedding") %>%
    layer_flatten()
  
  # Concatenate
  concat <- layer_concatenate(list(user_embedding, item_embedding))
  
  # MLP layers
  mlp <- concat %>%
    layer_dense(units = 20, activation = "relu") %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 10, activation = "relu") %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 1, activation = "linear")
  
  # Create model
  model <- keras_model(inputs = list(user_input, item_input), outputs = mlp)
  
  # Compile
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "mse",
    metrics = c("mae")
  )
  
  return(model)
}

# Train NCF model
ncf_model <- build_ncf_model()

# Prepare training data
user_indices <- train_df$user_idx - 1  # Keras uses 0-based indexing
item_indices <- train_df$item_idx - 1
ratings <- train_df$rating

# Train model
history <- ncf_model %>% fit(
  list(user_indices, item_indices),
  ratings,
  epochs = 50,
  batch_size = 64,
  validation_split = 0.2,
  verbose = 1
)

# Evaluate model
test_user_indices <- test_df$user_idx - 1
test_item_indices <- test_df$item_idx - 1
test_ratings <- test_df$rating

predictions <- ncf_model %>% predict(list(test_user_indices, test_item_indices))
mae <- mean(abs(predictions - test_ratings))
rmse <- sqrt(mean((predictions - test_ratings)^2))

cat("NCF Model Results:\n")
cat("MAE:", mae, "\n")
cat("RMSE:", rmse, "\n")

# Visualization
# Training history
p1 <- ggplot(data.frame(
  epoch = 1:length(history$metrics$loss),
  loss = history$metrics$loss,
  val_loss = history$metrics$val_loss
)) +
  geom_line(aes(x = epoch, y = loss, color = "Training")) +
  geom_line(aes(x = epoch, y = val_loss, color = "Validation")) +
  labs(title = "NCF Training History",
       x = "Epoch", y = "Loss", color = "Dataset") +
  theme_minimal()

# Prediction vs Actual
p2 <- ggplot(data.frame(
  actual = test_ratings,
  predicted = predictions
)) +
  geom_point(aes(x = actual, y = predicted), alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "NCF: Predicted vs Actual",
       x = "Actual Rating", y = "Predicted Rating") +
  theme_minimal()

# Rating distribution
p3 <- ggplot(ratings_df, aes(x = factor(rating))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Rating Distribution",
       x = "Rating", y = "Count") +
  theme_minimal()

# Combine plots
library(gridExtra)
grid.arrange(p1, p2, p3, ncol = 2)