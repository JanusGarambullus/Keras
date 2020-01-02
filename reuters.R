
# Load libraries ----------------------------------------------------------

library(tidyverse)
library(keras)

# Load data ---------------------------------------------------------------

reuters <- dataset_reuters(num_words = 5000)

c(c(train_x, train_y), c(test_x, test_y)) %<-% reuters

# Decode article ----------------------------------------------------------

word_index <- dataset_reuters_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
decoded_newswire <- sapply(train_x[[2]], function(index) {
  # Note that our indices were offset by 3 because 0, 1, and 2
  # are reserved indices for "padding", "start of sequence", and "unknown".
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})

cat(decoded_newswire)

# Process data ------------------------------------------------------------

# Garbage collection: gc() - use it to clean up memory after deleting large variable
# How much damn memory do these guys have?

# Vectorise sequences
vectorize_sequences <- function(sequences, dimension = 5000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

train_x <- vectorize_sequences(train_x)
test_x <- vectorize_sequences(test_x)

# One-hot encode
to_one_hot <- function(labels, dimension = 46) {
  results <- matrix(0, nrow = length(labels), ncol = dimension)
  for (i in 1:length(labels))
    results[i, labels[[i]] + 1] <- 1
  results
}

one_hot_train_labels <- to_one_hot(train_x)
one_hot_test_labels <- to_one_hot(test_x)


