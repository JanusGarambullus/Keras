
# Load libraries ----------------------------------------------------------

library(tidyverse)
library(keras)

# Load data ---------------------------------------------------------------

mnist <- dataset_mnist()

x_train <- mnist$train$x
y_train <- mnist$train$y

x_test <- mnist$test$x
y_test <- mnist$test$y

# Reshape features --------------------------------------------------------

mnist_reshape <- function(data) {
  array_reshape(data, c(nrow(data), 784))
}

x_train <- mnist_reshape(x_train)
x_test <- mnist_reshape(x_test)

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Defining the model ------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")

summary(model)

# Compile model -----------------------------------------------------------

model %>%
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy")
  )

# Training ----------------------------------------------------------------

history <- model %>%
  fit(
    x_train, y_train,
    epochs = 30, batch_size = 128,
    validation_split = 0.2
  )

plot(history)

# Evalutation -------------------------------------------------------------

model %>%
  evaluate(x_test, y_test)

model %>%
  predict_classes(x_test)

