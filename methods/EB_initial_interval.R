b <- 3
y <- seq(0, 2000, 1)
a1 <- c()
a2 <- c()
for (n in y) {
    L <- max(b, qgamma(0.05, n, 1)) - b
    U <- max(b, qgamma(0.95, n + 1, 1)) - b
    a1 <- c(a1, L)
    a2 <- c(a2, U)
}
matrix <- rbind(y, a1, a2)

# Transpose the matrix so that each observation occupies one row.
matrix <- t(matrix)

