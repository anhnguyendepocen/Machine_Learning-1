
library(RcppDL)

# Load data
my.data <- ISLR::Weekly
head(my.data,3)

my.data$Direction <- as.numeric (my.data$Direction)
head(my.data,3)

# Transform data for preparation
my.data$Direction <- my.data$Direction - 2
my.data[ my.data$Direction == 0 , "Direction"] <- 1
my.data$Direction2 <- - my.data$Direction
my.data <- my.data[, !(colnames(my.data) %in% c("Year", "Volume", "Today"))]
my.data <- sign(my.data)
my.data[my.data == -1] <- 0
head(my.data,3)

# Split data to train and test sets
x_train <- as.matrix(my.data[1:989,1:5])
x_test <- as.matrix(my.data[990:1089,1:5])
y_train <- as.matrix(my.data[1:989,6:7])
y_test <- as.matrix(my.data[990:1089,6:7])

# Specify the Deep Auto Encoder model
hidden <- c(12,12)
deep_auto <- RcppDL::Rsda(x_train, y_train, hidden)
RcppDL::setCorruptionLevel(deep_auto, x = 0.3)  # corruption level is for denoising
RcppDL::setFinetuneLearningRate(deep_auto, x = 0.1)
summary(deep_auto)

# Pretrain
pretrain(deep_auto)

# Finetuning
finetune(deep_auto)

# Make predictions
pred <- predict(deep_auto, x_test)
pred <- ifelse( pred[, 1] >= mean(y_train[,1]), 1, 0)
table(pred, y_test[,1])



