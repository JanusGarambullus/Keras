
# Load libraries ----------------------------------------------------------

library(tidyverse)
library(keras)

# Load data ---------------------------------------------------------------

mnist <- dataset_mnist()

c(c(train_x, train_y), c(test_x, test_y)) %<-% mnist

# Preprocess data ---------------------------------------------------------

# Reshape
train_x <- train_x %>%
  array_reshape(c(60000, 28, 28, 1))

test_x <- test_x %>%
  array_reshape(c(10000, 28, 28, 1))

# Normalise
train_x <- train_x / 255
test_x <- test_x / 255

# Label to categorical
train_y <- to_categorical(train_y)
test_y <- to_categorical(test_y)

# Build model -------------------------------------------------------------

# Specify layers
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

model <- model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

# Compile model
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Train model
model %>% fit(
  train_x, train_y, 
  epochs = 5, batch_size = 64,
  validation_split = 0.2
)

# Evaluate model ----------------------------------------------------------

# We are happy with the validated performance of our model. We can now
# compute the test set accuracy.

results <- model %>%
  evaluate(test_x, test_y)

results

# Jesus. These are amazing results. 99.2% accuracy! Convolutional neural networks 
# are really powerful.

