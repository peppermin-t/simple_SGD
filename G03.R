# Names and university usernames
# githublink
# contribution

# A function to build a list representing the network
# Input: d (a vector giving the number of nodes in each layer of a network)
# !!test: d <- c(3,4,4,2)
netup <- function(d) {
  # Number of layers
  n <- length(d)

  # Initialise a list to store the weight matrices (W) and offset vectors (b)
  h <- W <- b <- list()

  # Loop over the layers from 1 to n-1
  for (i in 1:(n - 1)) {
    # Create a list of nodes for each layer (h)
    h[[i]] <- rep(0, d[i])

    # Define dimensions for each link which is given by the number of nodes
    # in current layer (d[i+1]) and that in the previous layer (d[i])

    # Then the weight matrix for each link is given by:
    W[[i]] <- matrix(runif(d[i] * d[i + 1], 0, 0.2), d[i + 1], d[i])

    # And the offset vectors for each link is given by:
    b[[i]] <- runif(d[i], 0, 0.2)
  }

  h[[n]] <- rep(0, d[n])

  list(h = h, W = W, b = b)
}

nn <- netup(d)

# A function to return an updated network list
# Input: nn (a list of network returned from netup()) and
# inp (a vector of input values for the first layer)
# !! calculate remaining node values given inp
# !! test inp <- rnorm(d[1])
forward <- function(nn, inp) {
  W <- nn$W
  b <- nn$b
  h <- nn$h

  # Use eq(1)
  # Store values for the first layer
  h[[1]] <- inp

  # Loop over each remaining layer
  for (i in range(1:length(h) - 1)) {
    h[[i + 1]] <- W[[i]] * h[[i]] + b[[i]]
  }


  list(h = h, W = W, b = b)
}


# A function to compute the derivatives of the loss for a network
# Input: nn (a list of network returned from forward()) and k (output class)
backward <- function(nn, k) {
  # output class k - ?? is k an output representing the number of class

  # Extract values of the neural network
  W <- nn$W
  h <- nn$h

  # Create empty lists for derivatives
  dh <- dW <- db <- list()

  n <- length(h)

  # Compute the probability that the output variable is in class k
  # ?When index of node is not equal to that at each iteration
  dh[[n]] <- exp(h[n]) / sum(exp(h)) # ?? maybe h[[n]]

  # ?When index of node is equal to that at each iteration
  dh[[n]][k] <- dh[[n]][k] - 1

  ## Apply chain rule to compute derivatives of loss
  # Loop over each layer from the last layer (back-propagation)
  for (i in (n - 1):1) {
    # Store the node values at each layer
    d <- dh[[i + 1]]

    # Set to zero when node value is negative
    d[which(h[[i + 1]] <= 0)] <- 0

    # Compute derivative w.r.t. the nodes (dh)
    dh[[i]] <- t(W[i]) %*% d

    # Compute derivative w.r.t. the offsets (db)
    db[[i]] <- d

    # Compute derivative w.r.t. the weights (dW)
    dW[[i]] <- d %*% h[i]
  }

  list(dh = dh, dW = dW, db = db)
}


# A function to train the network given input data and labels
# Input: inp (a matrix of input data), k (a vector of labels), eta (step size),
# mb (number of randomly sampled data to compute gradient),
# nstep (number of optimization steps to take)
train <- function(nn, inp, k, eta = .01, mb = 10, nstep = 10000) {
  # Compute the number of layers for the network
  n <- length(inp)

  # Loop over each step
  for (i in 1:nstep) {
    # Determine the index of data used for gradient calculation
    iid <- sample(n, mb)

    # Extract randomly sampled data
    sub_inp <- inp[iid]

    # Extract its corresponding labels
    sub_k <- k[iid]

    # Initialise gradients

    # ? grads can be short for graduation, but here is gradient
    # ! Graduation: process of using statistical tech. to improve
    #   estimates provided by the crude rates
    grads <- c() # shall I use matrix calculation for this?

    # Loop over each sampled data
    for (i in 1:mb) { # use apply?

      # Compute the network list
      nn <- forward(nn, sub_inp) # n data runs parallel ? No

      # Compute gradients by taking derivatives of the loss
      grads <- c(grads, backward(nn, sub_k)) # n data runs parallel ? No
    }

    # Compute number of layers
    l <- length(nn$h)

    # Loop over each layer
    for (i in 1:l) {
      # Update weight parameters (W)
      nn$W[[i]] <- nn$W[[i]] - eta * mean(grads)$dW

      # Update offset parameters (b)
      nn$b[[i]] <- nn$b[[i]] - eta * mean(grads)$db
    }
  }

  nn
}


# A function to train a network to classify items to groups based on a given
# output class k
# Input:nn (a list of network returned from forward()),
# inp (a matrix of input data), k (a vector of labels).
test <- function(nn, inp, k) {
  # Compute number of labels
  n <- length(k)

  # Compute number of layers
  l <- length(nn$h)

  # Initialise number of misclassificated data
  mis_class <- 0

  # Loop over each output class / label
  for (i in 1:n) {
    # Extract node values using transformation (ReLU)
    output_h <- forward(nn, inp[i, ])$h[[l]]

    # Compute the probability that the output variable is labelled k / in output class k
    scores <- exp(output_h) / sum(exp(output_h))

    # Find the most frequently appeared probability
    k_ <- which(scores == max(scores))

    # Count the misclassified data; occur when output is not the maximum
    if (k_ != k[i]) {
      mis_class <- mis_class + 1
    }
  }

  # Compute the probability of loss
  mis_class / n
}

# ! iris  # Sepal.Length, Sepal.Width, Petal.Length, Petal.Width, Species
# Aim: Train a network to classify irises to species based on given characteristics

# k <- 4 # output class

# Define the dimensions of the neural net
d <- c(4, 8, 7, 3)

# Create a network based on the dimensions defined
nn <- netup(d)

# data preparation
features <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")
label <- c("Species")
label_options <- c("setosa", "versicolor", "virginica")

inp <- as.matrix(iris[, features])
k <- as.integer(factor(iris[, label], level = label_options))

## Divide data into test data and training data

# Test data contains every 5th row of iris dataset
# test_data <- iris[seq(5, nrow(iris), by = 5),]
test_iid <- seq(5, nrow(iris), by = 5)

# The rest is training data
# train_data <- iris[-seq(5, nrow(iris), by = 5),]
inp_train <- inp[-test_iid]
k_train <- k[-test_iid]
inp_test <- inp[test_iid]
k_test <- k[test_iid]

# set.seed() # training has worked

# Classify test data to species according to the class predicted
# ?? plot
# Compute misclassification rate
mis_class_rate <- test(nn, inp_test, k_test)
