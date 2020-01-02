
# Load libraries ----------------------------------------------------------

library(tidyverse)
library(keras)

# Load data ---------------------------------------------------------------

imdb <- dataset_imdb(num_words = 10000)

train_x <- imdb$train$x
train_y <- imdb$train$y

test_x <- imdb$test$x
test_y <- imdb$test$y

# Decode to text ----------------------------------------------------------

word_index <- dataset_imdb_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

decoded_review <- sapply(train_x[[1]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})

cat(decoded_review)

# Data preparation --------------------------------------------------------

vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences)) 
    results[i, sequences[[i]]] <- 1
  
  results
}

train_x <- vectorize_sequences(train_x)
test_x <- vectorize_sequences(test_x)

train_y <- as.numeric(train_y)
test_y <- as.numeric(test_y)

# Create validation set
val_indices <- 1:10000
x_val <- train_x[val_indices,]
partial_x_train <- train_x[-val_indices,]
y_val <- train_y[val_indices]
partial_y_train <- train_y[-val_indices]

# Model building ----------------------------------------------------------

model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )

# With validation
history <- model %>%
  fit(train_x,
      train_y,
      epochs = 10,
      batch_size = 512,
      validation_split = 0.2)

plot(history)

# Evaluate test set predictions -------------------------------------------

results <- model %>% evaluate(test_x, test_y)

# Make predictions --------------------------------------------------------

pred <- model %>%
  predict(test_x)

head(pred)
