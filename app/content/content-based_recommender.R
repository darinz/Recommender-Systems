# Content-Based Recommender System in R
library(tm)
library(proxy)
library(ggplot2)
library(dplyr)
library(tidyr)

content_based_recommender <- function(similarity_metric = "cosine") {
  list(
    similarity_metric = similarity_metric,
    item_profiles = NULL,
    user_profiles = NULL,
    feature_names = NULL
  )
}

compute_similarity <- function(profile1, profile2, metric = "cosine") {
  if (metric == "cosine") {
    return(sum(profile1 * profile2) / (sqrt(sum(profile1^2)) * sqrt(sum(profile2^2))))
  } else if (metric == "euclidean") {
    distance <- sqrt(sum((profile1 - profile2)^2))
    return(1 / (1 + distance))
  } else if (metric == "pearson") {
    return(cor(profile1, profile2, method = "pearson"))
  }
}

create_item_profiles <- function(recommender, items_df, feature_columns, text_columns = NULL) {
  profiles <- list()
  feature_names <- c()
  
  # Handle categorical features
  for (col in feature_columns) {
    if (is.character(items_df[[col]])) {
      # Encode categorical features
      unique_values <- unique(items_df[[col]])
      encoded_matrix <- matrix(0, nrow = nrow(items_df), ncol = length(unique_values))
      
      for (i in 1:length(unique_values)) {
        encoded_matrix[items_df[[col]] == unique_values[i], i] <- 1
      }
      
      profiles[[length(profiles) + 1]] <- encoded_matrix
      feature_names <- c(feature_names, paste0(col, "_", unique_values))
    } else {
      # Numerical features
      profiles[[length(profiles) + 1]] <- matrix(items_df[[col]], ncol = 1)
      feature_names <- c(feature_names, col)
    }
  }
  
  # Handle text features
  if (!is.null(text_columns)) {
    for (col in text_columns) {
      # Create corpus
      corpus <- Corpus(VectorSource(items_df[[col]]))
      
      # Create document-term matrix
      dtm <- DocumentTermMatrix(corpus, control = list(
        removePunctuation = TRUE,
        removeNumbers = TRUE,
        stopwords = TRUE,
        weighting = weightTfIdf
      ))
      
      # Convert to matrix
      text_matrix <- as.matrix(dtm)
      
      # Limit features
      if (ncol(text_matrix) > 50) {
        text_matrix <- text_matrix[, 1:50]
      }
      
      profiles[[length(profiles) + 1]] <- text_matrix
      feature_names <- c(feature_names, paste0(col, "_", colnames(text_matrix)))
    }
  }
  
  # Combine all features
  recommender$item_profiles <- do.call(cbind, profiles)
  recommender$feature_names <- feature_names
  
  # Normalize features
  recommender$item_profiles <- scale(recommender$item_profiles)
  
  return(recommender)
}

create_user_profiles <- function(recommender, ratings_df, items_df) {
  user_profiles <- list()
  
  for (user_id in unique(ratings_df$user_id)) {
    user_ratings <- ratings_df[ratings_df$user_id == user_id, ]
    
    # Get items rated by this user
    rated_items <- user_ratings$item_id
    ratings <- user_ratings$rating
    
    # Find corresponding item profiles
    item_indices <- match(rated_items, items_df$movie_id)
    item_profiles <- recommender$item_profiles[item_indices, ]
    
    # Compute weighted average (weighted by ratings)
    weights <- ratings / sum(ratings)
    user_profile <- colSums(t(item_profiles) * weights)
    
    user_profiles[[as.character(user_id)]] <- user_profile
  }
  
  recommender$user_profiles <- user_profiles
  return(recommender)
}

recommend <- function(recommender, user_id, n_recommendations = 5) {
  if (!(as.character(user_id) %in% names(recommender$user_profiles))) {
    return(list())
  }
  
  user_profile <- recommender$user_profiles[[as.character(user_id)]]
  
  # Compute similarities with all items
  similarities <- sapply(1:nrow(recommender$item_profiles), function(i) {
    compute_similarity(user_profile, recommender$item_profiles[i, ], recommender$similarity_metric)
  })
  
  # Sort by similarity
  sorted_indices <- order(similarities, decreasing = TRUE)
  
  # Return top recommendations
  result <- list()
  for (i in 1:n_recommendations) {
    result[[i]] <- list(
      item_index = sorted_indices[i],
      similarity = similarities[sorted_indices[i]]
    )
  }
  
  return(result)
}

# Generate synthetic data
set.seed(42)
n_movies <- 100
n_users <- 50

# Create movie features
movies_df <- data.frame(
  movie_id = 1:n_movies,
  title = paste0("Movie_", 1:n_movies),
  genre = sample(c("Action", "Drama", "Comedy", "Thriller", "Romance"), n_movies, replace = TRUE),
  year = sample(1990:2023, n_movies, replace = TRUE),
  rating = runif(n_movies, 1, 10),
  budget = runif(n_movies, 1, 100),
  director = sample(c("Spielberg", "Nolan", "Tarantino", "Scorsese", "Cameron"), n_movies, replace = TRUE),
  description = paste0("Description for movie ", 1:n_movies)
)

# Create synthetic ratings
ratings_data <- list()
for (user_id in 1:n_users) {
  n_ratings <- sample(5:20, 1)
  rated_movies <- sample(1:n_movies, n_ratings, replace = FALSE)
  
  for (movie_id in rated_movies) {
    movie <- movies_df[movie_id, ]
    base_rating <- 5
    
    # Genre preferences
    if (movie$genre %in% c("Action", "Thriller")) {
      base_rating <- base_rating + rnorm(1, 1, 1)
    } else if (movie$genre %in% c("Drama", "Romance")) {
      base_rating <- base_rating + rnorm(1, -1, 1)
    }
    
    # Year preference
    year_factor <- (movie$year - 1990) / (2023 - 1990)
    base_rating <- base_rating + year_factor * 2
    
    # Add noise
    rating <- max(1, min(10, base_rating + rnorm(1, 0, 1)))
    
    ratings_data[[length(ratings_data) + 1]] <- list(
      user_id = user_id,
      movie_id = movie_id,
      rating = rating
    )
  }
}

ratings_df <- do.call(rbind, lapply(ratings_data, as.data.frame))

# Initialize and train recommender
recommender <- content_based_recommender("cosine")
feature_columns <- c("genre", "year", "rating", "budget", "director")
text_columns <- c("description")

recommender <- create_item_profiles(recommender, movies_df, feature_columns, text_columns)
recommender <- create_user_profiles(recommender, ratings_df, movies_df)

# Generate recommendations
test_user <- 1
recommendations <- recommend(recommender, test_user, 10)

cat("Top 10 recommendations for User", test_user, ":\n")
for (i in 1:length(recommendations)) {
  item_idx <- recommendations[[i]]$item_index
  similarity <- recommendations[[i]]$similarity
  movie <- movies_df[item_idx, ]
  cat(sprintf("%d. %s (%s, %d) - Similarity: %.3f\n", 
              i, movie$title, movie$genre, movie$year, similarity))
}