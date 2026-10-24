# Define the functions.
f <- function(u, n) ppois(n, b + 0) - 0.9 * ppois(n, b) - ppois(n, b + u)
f1 <- function(ll, n) (1 - ppois(n, b + ll)) / ppois(n, b)
f2 <- function(uu, n) (1 - ppois(n, b + uu)) / ppois(n, b)
g <- function(lambda, n) dpois(n, b + lambda) / ppois(n, b)
g1 <- function(lambda, c, n) dpois(n, b + lambda) / ppois(n, b) - c
g2 <- function(lambda, c, n) dpois(n, b + lambda) / ppois(n, b) - c

# Set model parameters.
b <- 3
results <- data.frame(n = integer(), lower = numeric(), upper = numeric()) # Store the resulting intervals.

# Iterate over the observed count n.
for (n in 0:75) {
  # Solve for z.
  z <- uniroot(f, c(-1, 1000), n = n)$root
  
  # Locate the value of lambda at which g is maximized.
  aa <- seq(0, 1000, 0.001)
  a <- sapply(aa, function(i) g(i, n))
  lc <- which.max(a)
  mod <- aa[lc]
  
  if (g(0, n) >= g(z, n)) {
    u <- z
    l <- 0
  } else {
    c0 <- g(0, n)
    c1 <- g(z, n)
    diff <- 1
    while (diff >= 0.001) {
      c <- (c0 + c1) / 2
      y <- uniroot(g1, c(0, mod), c = c, n = n)$root
      zz <- uniroot(g2, c(mod, 1000), c = c, n = n)$root
      diff <- f2(zz, n) - f1(y, n) - 0.9
      if (diff <= 0) {
        c1 <- c
      } else {
        c0 <- c
      }
    }
    l <- y
    u <- zz
  }
  # Store the interval endpoints.
  results <- rbind(results, data.frame(n = n, lower = l, upper = u))
}

# Print the results.
print(results)
