f <- function(theta, w1, y, u) {
  w1 * pgamma(theta, y, 1) + (1 - w1) * pgamma(theta, y + 1, 1) - u 
}

results <- data.frame( y=integer(), IML=numeric(), IMU=numeric())

# Generate w and u values for w1, u and w2, v
len_wU0 <- 30
wU0_1 <- vector()
for (i in 0:(len_wU0 - 1)) {
  wU0_1 <- c(wU0_1, i / len_wU0 + 1 / (2 * len_wU0))
}

w_u_combinations <- expand.grid(w=wU0_1, u=wU0_1)

 
for (y in 0:2000) {   
  c <- vector()    
  for (j in 1:nrow(w_u_combinations)) {
    w1 <- w_u_combinations$w[j]
    u <- w_u_combinations$u[j]
    
    a1 <- uniroot(f, c(-1, 10000), w1 = w1, y = y, u = u)$root
    
    c <- c(c, a1)      
  }    
  IM1 <- unname(quantile(c, 0.05))
  IM2 <- unname(quantile(c, 0.95))
  IML <- max(3, IM1)
  IMU <- max(3, IM2)  

  results <- rbind(results, data.frame(y = y, IML = IML, IMU = IMU))
}

