# EA Project: Predicting FIFA 16 Sales - Sheng Zhang

# Read in data
fifa_raw <- read.csv("./data_stack.csv",header = TRUE)

# Examine dataset
head(fifa_raw)
summary(fifa_raw)
fifa_raw[fifa_raw$run_on==0,]
fifa <- fifa_raw[fifa_raw$run_on!=0,]
summary(fifa)

# Process geodata
library(zipcode)
library(maps)
library(viridis)
library(ggplot2)
require(scales)
fifa$zip <- clean.zipcodes(fifa$postal_code)
data(zipcode)
fifa <- merge(fifa, zipcode, by = 'zip')
table(fifa$state)
qplot(as.factor(fifa$state), xlab="State", ylab="") + scale_y_continuous(labels = comma)
usa <- map_data('state')
ggplot(fifa,aes(longitude,latitude)) +
  geom_polygon(data=usa,aes(x=long,y=lat,group=group),color='gray',fill=NA,alpha=.35)+
  geom_point(size=.15,alpha=.25) +
  xlim(-125,-65)+ylim(20,50)

# Compute FIFA popularity by state
aggregate(is.na(fifa[,7:13]), list(fifa$state), mean)

# Calculate important descriptive data
f15 <- 1 - sum(is.na(fifa$F15))/nrow(fifa)
f14 <- 1 - sum(is.na(fifa$F14))/nrow(fifa)
f13 <- 1 - sum(is.na(fifa$F13))/nrow(fifa)
f12 <- 1 - sum(is.na(fifa$F12))/nrow(fifa)
f11 <- 1 - sum(is.na(fifa$F11))/nrow(fifa)
f10 <- 1 - sum(is.na(fifa$F10))/nrow(fifa)
f09 <- 1 - sum(is.na(fifa$F09))/nrow(fifa)
PS3 <- sum(fifa$PS3 == 1)/nrow(fifa)
PS4 <- sum(fifa$PS4 == 1)/nrow(fifa)
X360 <- sum(fifa$X360 == 1)/nrow(fifa)
XONE <- sum(fifa$XONE == 1)/nrow(fifa)
PC <- sum(fifa$PC == 1)/nrow(fifa)

# Visualize important descriptive data
qplot(fifa$day_cnt, xlab="Days_Active", ylab="")  + scale_y_continuous(labels = comma)
qplot(log(fifa$day_cnt), xlab="Log_of_Days_Active", ylab="")  + scale_y_continuous(labels = comma)
qplot(as.factor(fifa$all_cnt), xlab="EA_Titles_Owned", ylab="")  + scale_y_continuous(labels = comma)
qplot(fifa$all_cnt, xlab="EA_Titles_Owned", ylab="")  + scale_y_continuous(labels = comma)
qplot(log(fifa$all_cnt), xlab="Log_of_EA_Titles_Owned", ylab="")  + scale_y_continuous(labels = comma)
qplot(as.factor(fifa$fifa_cnt), xlab="FIFA_Titles_Owned", ylab="")  + scale_y_continuous(labels = comma)

# Prepare datasets for learning
fifa$active_rate <- fifa$day_cnt/365
fifa$ea_titles_over_max <- fifa$all_cnt/max(fifa$all_cnt)
fifa$fifa_titles_over_max <- fifa$fifa_cnt/max(fifa$fifa_cnt)
fifa_features <- fifa[,c(7:18,20,23:25)]
summary(fifa_features)
fifa_features[,1:7] <- ifelse(is.na(fifa_features[,1:7]),0,1)
fifa_features$state <- as.factor(fifa_features$state)
fifa_features <- as.matrix(sapply(fifa_features,as.numeric))
fifa_features[,8] <- ifelse(fifa_features[,8]==3,0,1)
summary(fifa_features)
fifa_15 <- fifa_features[,1:16]
fifa_16 <- fifa_features[,c(1:6,8:16)]
# Split data used for predicting FIFA 15 sales into training and test sets
set.seed(1)
fifa.train.id <- sample(1:nrow(fifa_features), 0.8*nrow(fifa_features))
fifa_15.train <- fifa_15[fifa.train.id, ]
fifa_15.test <- fifa_15[-fifa.train.id, ]
x_train <- fifa_15.train[,2:16]
y_train <- fifa_15.train[,1]
x_test <- fifa_15.test[,2:16]
y_test <- fifa_15.test[,1]

# Random Forest
library(randomForest)
library(MASS)
set.seed(2)
rf.fifa <- randomForest(x_train, y = y_train, ntree = 10, mtry = 5, importance = TRUE)
importance(rf.fifa)
rf.pred_prob_15 <- predict(rf.fifa, x_test, type="class")
rf.pred_15 <- ifelse(rf.pred_prob_15 >= mean(y_train), 1, 0)
summary(rf.pred_15)
rf.table <- table(rf.pred_15, y_test)
rf.accuracy <- (rf.table[1,1]+rf.table[2,2])/sum(rf.table)
rf.accuracy
# Predict FIFA 16 adoption with the model
rf.pred_prob_16 <- predict(rf.fifa, fifa_16, type="class")
summary(rf.pred_prob_16)
qplot(rf.pred_prob_16, xlab="Predicted Purchase Likelihood", ylab="") + scale_y_continuous(labels = comma)

# Logistic regression (Lasso)
library(glmnet)
grd <- 10 ^ seq(10,-1,length=100)
lasso.fifa <- glmnet(x_train, y_train, alpha=1,lambda=grd,family="binomial")
coef(lasso.fifa)
set.seed(1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, nfolds=3, family="binomial")
plot(cv.out)
bestlam <- cv.out$lambda.min
out <- glmnet(x_train, y_train, nlambda=1, lambda=bestlam, alpha=1, family="binomial")
coef(out)
lasso.pred_prob_15 <- predict(out,s=bestlam,x_test,type="response")
lasso.pred_15 <- ifelse(lasso.pred_prob_15[, 1] >= mean(y_train), 1, 0)
summary(lasso.pred_15)
lasso.table <- table(lasso.pred_15, y_test)
lasso.accuracy <- (lasso.table[1,1]+lasso.table[2,2])/sum(lasso.table)
lasso.accuracy
# Predict FIFA 16 adoption with the model
lasso.pred_prob_16 <- predict(out, s=bestlam, fifa_16,type="response")
summary(lasso.pred_prob_16)
qplot(lasso.pred_prob_16, xlab="Predicted Purchase Likelihood", ylab="") + scale_y_continuous(labels = comma)

