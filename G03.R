# Names and university usernames
# githublink
# contribution

# A function to build a list representing the network
# Input: d (a vector giving the number of nodes in each layer of a network)

netup <- function(d) {
  # Number of layers
  n <- length(d)

  # Initialise a list to store the weight matrices (W) and offset vectors (b)
  h <- W <- b <- list()

  # Loop over the layers from 1 to n-1
  for (i in 1: (n - 1)) {
    # Create a list of nodes for each layer (h)
    h[[i]] <- rep(0, d[i])

    # Define dimensions for each link which is given by the number of nodes
    # in current layer (d[i+1]) and that in the previous layer (d[i])

    # Then the weight matrix for each link is given by:
    W[[i]] <- matrix(runif(d[i] * d[i + 1], 0, 0.2), d[i + 1], d[i])

    # And the offset vectors for each link is given by:
    b[[i]] <- runif(d[i + 1], 0, 0.2)
  }

  h[[n]] <- rep(0, d[n])

  list(h = h, W = W, b = b)
}

# A function to return an updated network list
# Input: nn (a list of network returned from netup()) and
# inp (a vector of input values for the first layer)
# !! calculate remaining node values given inp
# !! test inp <- rnorm(d[1])
forward <- function(nn, inp) {
  W <- nn$W
  b <- nn$b
  h <- nn$h

  # Obtain the number of layers in nn
  l <- length(h)

  # Use eq(1)
  # Store values for the first layer
  h[[1]] <- inp

  # Loop over each remaining layer
  for (i in 1: (l - 1)) {
    # Compute transformation to obtain node values and
    #   set equal to zero if element is negative
    h[[i + 1]] <- pmax(drop(W[[i]] %*% h[[i]] + b[[i]]), 0)
  }

  list(h = h, W = W, b = b)
}


# A function to compute the derivatives of the loss for a network
# Input: nn (a list of network returned from forward()) and
#   k (the real output label k for this run)
backward <- function(nn, k) {

  # Obtain values of the neural network
  W <- nn$W
  h <- nn$h

  # Create empty lists for derivatives
  dh <- dW <- db <- list()

  l <- length(h)

  # Compute the probability that the output variable is in class k
  # ?When index of node is not equal to that at each iteration
  dh[[l]] <- exp(h[[l]]) / sum(exp(h[[l]]))

  # ?When index of node is equal to that at each iteration
  dh[[l]][k] <- dh[[l]][k] - 1

  ## Apply chain rule to compute derivatives of loss
  # Loop over each layer from the last layer (back-propagation)
  for (i in (l - 1): 1) {
    # Store the node values at each layer
    d <- dh[[i + 1]]

    # Set to zero when node value is negative
    d[which(h[[i + 1]] <= 0)] <- 0

    # Compute derivative w.r.t. the nodes (dh)
    dh[[i]] <- t(W[[i]]) %*% d

    # Compute derivative w.r.t. the offsets (db)
    db[[i]] <- d

    # Compute derivative w.r.t. the weights (dW)
    dW[[i]] <- d %*% t(h[[i]])
  }

  c(nn, list(dh = dh, dW = dW, db = db))
}


# A function to train the network given input data and labels
# Input: inp (a matrix of input data), k (a vector of labels), eta (step size),
# mb (number of randomly sampled data to compute gradient),
# nstep (number of optimization steps to take)
train <- function(nn, inp, k, eta = .01, mb = 10, nstep = 10000) {
  # Obtain the number of train data
  n <- length(k)

  # Obtain the number of layers for the network
  l <- length(nn$h)

  # losses for each batch in a step, only for recording
  losses <- c()

  # Loop over each step
  for (i in 1: nstep) {
    # Determine the index of data used for gradient calculation
    iid <- sample(n, mb)

    # Extract randomly sampled data
    sub_inp <- inp[iid, ]

    # Extract its corresponding labels
    sub_k <- k[iid]

    # scores for each run in a batch, only for recording
    scores <- c()

    # Initialise gradients
    grads_W <- grads_b <- list()

    # Obtain the size of each layer
    d <- sapply(nn$h, length)

    for (i in 1: (l - 1)) {
      # Then the weight matrix for each link is given by:
      grads_W[[i]] <- matrix(rep(0, d[i] * d[i + 1]), d[i + 1], d[i])

      # And the offset vectors for each link is given by:
      grads_b[[i]] <- rep(0, d[i + 1])
    }

    # Loop over each sampled data
    for (i in 1: mb) { # use apply?

      # Forward the input data into the network and get the nn updated with h
      nn <- forward(nn, sub_inp[i, ]) # n data runs parallel ? No

      # Obtain the output vector form the updated nn
      output_h <- nn$h[[l]]

      # Compute the probability that the output variable is
      #   labelled k / in output class k
      scores <- c(scores, log(exp(output_h[sub_k[i]]) / sum(exp(output_h))))

      # Compute gradients with gradient descent
      grad <- backward(nn, sub_k[i])

      for (i in 1: (l - 1)) {
        grads_W[[i]] <- grads_W[[i]] + grad$dW[[i]]  # n data runs parallel ? No
        grads_b[[i]] <- grads_b[[i]] + grad$db[[i]]  # n data runs parallel ? No
      }
    }
    losses <- c(losses, - mean(scores))

    grad_W <- sapply(grads_W, function(x) x / mb)
    grad_b <- sapply(grads_b, function(x) x / mb)

    # Loop over each layer
    for (i in 1: (l - 1)) {
      # Update weight parameters (W)
      nn$W[[i]] <- nn$W[[i]] - eta * grad_W[[i]]

      # Update offset parameters (b)
      nn$b[[i]] <- nn$b[[i]] - eta * grad_b[[i]]
    }
  }

  # # plotting loss against step, only for recording
  # xs <- seq(1, nstep, by=5)
  # ys <- losses[xs]
  # plot(xs, ys, col="yellow", pch=20)
  
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
  for (i in 1: n) {
    # Extract node values using transformation (ReLU)
    output_h <- forward(nn, inp[i, ])$h[[l]]
    # print(output_h)

    # Compute the probability that the output variable is
    #   labelled k / in output class k
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

# set.seed(57) # training has worked

# Aim: Train a network to classify irises to species based on given features
# Define the dimensions of the neural network
d <- c(4, 8, 7, 3)

# data preparation
features <- colnames(iris)[1: 4]
label <- colnames(iris)[5]
label_options <- unique(iris[, 5])

inp <- as.matrix(iris[, features])
k <- as.integer(factor(iris[, label], level = label_options))

## Divide data into test data and training data
# Test data contains every 5th row of iris dataset
test_iid <- seq(5, nrow(iris), by = 5)

inp_train <- inp[-test_iid, ]
k_train <- k[-test_iid]
inp_test <- inp[test_iid, ]
k_test <- k[test_iid]

# Create a network based on the dimensions defined
nn <- netup(d)

# train nn
system.time(nn <- train(nn, inp_train, k_train, eta=.01, mb=10, nstep=10000))

# Classify test data to species according to the class predicted
#  and compute misclassification rate
# ?? plot
system.time(mis_class_rate <- test(nn, inp_test, k_test))
print(mis_class_rate)

# do grad need to return W h b?
