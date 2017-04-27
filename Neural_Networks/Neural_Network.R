
# Generate asset return data
asset <- data.frame(matrix(10,2520,1))
mu <- 0.05; sigma <- 0.6
for (i in 2:2520)
{
  asset[i,1] <- asset[i-1,1] * exp( rnorm(1, mu /252, sigma ^2 /252))
}

# Plot the trend
plot( asset[, 1], type="l")

max_asset <- max(asset)
min_asset <- min(asset)
strikes <- seq(min_asset * 0.5, max_asset * 1.5, length.out = 100)
plot(strikes)

# Construct option prices
options <- data.frame(matrix(NA, dim(asset)[1], 3))
options[, 1] <- sample(strikes, size = dim(asset)[1], replace = TRUE)
options[, 2] <- asset[sample(dim(asset)[1]), ]
rtrn <- na.omit(timeSeries::returns(asset[,1]))
sigma_est <- sqrt(sd(rtrn[,1]) * 252)
mu_est <- mean(rtrn[,1]) * 252
for (i in 1:dim(options)[1])
{
  BS <- fOptions::GBSOption(TypeFlag = "c", S = options[i, 2],
                            X = options[i, 1], Time = 1/4,
                            r = 0.08, b = 0, sigma = sigma_est)
  options[i, 3] <- BS@price
  
}

# Visualize the option returns
scatterplot3d::scatterplot3d(x = options[,1], y =options[,2], z = options[,3], pch = 16, type="h")

# Build the neural network
set.seed(1)
my_sample <- sample.int(nrow(options), round(.75*nrow(options)), replace = F)
train_options <- options[my_sample, ]
test_options <- options[-my_sample, ]
y_train <- scale(train_options[,3])
x_train <- scale(train_options[,1:2])
y_test <- scale(test_options[,3])
x_test <- scale(test_options[,1:2])
net2 <- RSNNS::mlp(x=x_train,y=y_train,size = c(10,10),
                   maxit = 1000, learnFuncParams = 0.01, linOut = TRUE )
