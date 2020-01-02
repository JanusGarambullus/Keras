
# Load libraries ----------------------------------------------------------

library(tidyverse)
library(keras)

# Load data ---------------------------------------------------------------

fashion_mnist <- dataset_fashion_mnist()

train_x <- fashion_mnist$train$x
train_y <- fashion_mnist$train$y

test_x <- fashion_mnist$test$x
test_y <- fashion_mnist$test$y

class_names <- c('T-shirt/top',
                 'Trouser',
                 'Pullover',
                 'Dress',
                 'Coat', 
                 'Sandal',
                 'Shirt',
                 'Sneaker',
                 'Bag',
                 'Ankle boot')

# Sense check -------------------------------------------------------------

# Create an image
image_1 <- as.data.frame(train_x[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- row(image_1)
image_1 <- image_1 %>% gather(key = "x", value = "value", -y)
image_1$x <- as.numeric(as.character(image_1$x))
image_1 <- as.data.frame(image_1)
image_1 <- image_1[1:nrow(image_1),]

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

# Preprocessing -----------------------------------------------------------

# Values need to be scaled for neural networks - the max value is 255
train_x <- train_x / 255
test_x <- test_x / 255

# Visualise ---------------------------------------------------------------

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- train_x[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_y[i] + 1]))
}

# Build model -------------------------------------------------------------

# Specify model type
model <- keras_model_sequential()

# Specify model
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

# Compile model
model %>%
  compile(optimizer = "adam",
          loss = "sparse_categorical_crossentropy",
          metrics = "accuracy")

# Train model
model %>%
  fit(train_x, train_y, epochs = 5)

# Evaluation --------------------------------------------------------------

score <- model %>% evaluate(test_x, test_y)

cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")

# The training accuracy mirrors the test set accuracy. Nice.

# Make predictions --------------------------------------------------------
 
predictions <- model %>%
  predict(test_x)

predictions[1, ]

# First few predictions
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- test_x[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_y[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}













