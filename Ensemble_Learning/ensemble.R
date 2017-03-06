install.packages("readr")
library(readr)
GDPC1 <- read_csv("~/Spring 2017/Machine Learning/GDPC1.csv")
## Parsed with column specification:
## cols(
## DATE = col date(format = ""),
## GDPC1 = col double()
## )
GDP <- as.data.frame(GDPC1)
head(GDPC1,3)

GDP$growth <- c(NA, diff(log(GDPC1$GDPC1)) * 100)
GDP <- na.omit(GDP)
head(GDP,3)

dim(GDP)
GDP.train <- GDP[1:269, ]
GDP.test <- GDP[270:279, ]

arma_1_1 <- arima(x = GDP.train$growth, order = c(1, 0, 1))
arma_1_1_predict <- predict( arma_1_1, n.ahead = 10, se.fit = FALSE)
arma_1_1_MSE <- mean( (arma_1_1_predict - GDP.test$growth) ^ 2 )
round(arma_1_1_MSE, 4)


# Ensemble for 15AR, 15MA, 4ARMA models

for (i in 1:15) 
{
data <- GDP.train[sample.int(200,1):269,]$growth
temp <- arima(x = data, order = c(i, 0, 0))
temp_predict <- predict( temp, n.ahead = 10, se.fit = FALSE)
ifelse ( exists("ensemble"), ensemble <- rbind( ensemble, temp_predict), ensemble <- temp_predict )
}

for (i in 1:15) 
{
data <- GDP.train[sample.int(200,1):269,]$growth
temp <- arima(x = data, order = c(0, 0, i))
temp_predict <- predict( temp, n.ahead = 10, se.fit = FALSE)
ensemble <- rbind( ensemble, temp_predict)
}

for (i in 1:2)
{ 
  for(j in 1:2 )
  {
    data <- GDP.train[sample.int(200,1):269,]$growth
    temp <- arima(x = data, order = c(i, 0, j))
    temp_predict <- predict( temp, n.ahead = 10, se.fit = FALSE)
    ifelse ( exists("ensemble"), ensemble <- rbind( ensemble, temp_predict), ensemble <- temp_predict )
  }
}


ensemble_predict <- colMeans(ensemble)
ensemble_MSE <- mean( (ensemble_predict - GDP.test$growth) ^ 2 )
round(ensemble_MSE, 4)
