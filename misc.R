
# Load libraries ----------------------------------------------------------

library(tidyverse)
library(keras)

# Create data -------------------------------------------------------------

x <- matrix(c(0, 1, 2, 3, 4, 5),
            nrow = 3, ncol = 2, byrow = T)

# Transpose matrix
t(x)

# Reshape matrix
res <- array_reshape(x, dim = c(6, 1))
res_1 <- array_reshape(x, dim = c(2, 3))

# These two are not the same!
res_1
t(x)

# Testing -----------------------------------------------------------------

grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

