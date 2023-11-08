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

  # # Loop over each layer
  # for (i in 1:n) {

  #   # Create a list of nodes for each layer (h)
  #   h[[i]] <- rep(0, d[i])
  # }
  
  # Loop over the layers from 1 to n-1
  for (i in 1: (n - 1)) {

    h[[i]] <- rep(0, d[i])

    # Define dimensions for each link which is given by the number of nodes 
    # in each layer (d[i]) and that in the next layer (d[i+1])

    # Then the weight matrix for each link is given by:
    W[[i]] <- matrix(runif(d[i + 1] * d[i], 0, 0.2), d[i + 1], d[i])

    # And the offset vectors for each link is given by:
    b[[i]] <- runif(d[i + 1], 0, 0.2)

  }

  h[[n]] <- rep(0, d[n])

  list(h=h, W=W, b=b)
}


# A function to return an updated network list 
# Input: nn (a list of network returned from netup()) and 
# inp (a vector of input values for the first layer)
# !! calculate remaining node values given inp
# !! test inp <- rnorm(d[1])
forward <- function(nn, inp) {
  # Use eq(1)
  # Store values for the first layer
  nn$h[[1]] <- inp
  
  
  
  
  list(updated_nn)
}

Lh <- function(h, k) {
  raw <- exp(h) / sum(exp(h))
  raw[k] <- raw[k] - 1
  raw
}


# A function to compute the derivatives of the loss for a network
# Input: nn (a list of network returned from forward()) and k (output class)
backward <- function(nn, k) {
  # output class k

  W <- nn$W
  b <- nn$b
  h <- nn$h

  dh <- dW <- db <- list()

  n <- length(h)

  dh[[n]] <- exp(h[n]) / sum(exp(h))
  dh[[n]][k] <- dh[[n]][k] - 1

  for (i in (n - 1): 1) {

    d <- dh[[i + 1]]
    d[which(h[[i + 1]] <= 0)] <- 0

    dh[[i]] <- t(W[i]) %*% d
    db[[i]] <- d
    dW[[i]] <- d %*% h[i]
  }
  # ??Loop over each layer, and loop over each nodes?
  
  # Differentiate w.r.t. the nodes (dh)
  
  # Differentiate w.r.t. the weights (dW)
  
  # Differentiate w.r.t. the offsets
  
  list(dh=dh, dW=dW, db=db)
}


# A function to train the network given input data and labels
# Input: inp (a matrix of input data), k (a vector of labels), eta (step size),
# mb (number of randomly sampled data to compute gradient), 
# nstep (number of optimization steps to take)
train <- function(nn, inp, k, eta=.01, mb=10, nstep=10000){
  #???????????
}


# Aim: Train a network to classify irises to species based on given characteristics
# d <- c(4,8,7,3) # network
# k <- 4 # output class?
# Divide data:
# - Test data contains every 5th row of iris dataset
# test_data <- iris[seq(5, nrow(iris), by = 5),]
# - The rest is training data
# train_data <- iris[-seq(5, nrow(iris), by = 5),]
# set.seed() # training has worked

# Classify test data to species according to the class predicted
# ?? plot
# Compute misclassification rate