
# Read in data
credit <- read.csv("./default of credit card clients.csv",header = TRUE)

# Transform data (use cumulative debt/credit variables)
cum_payment <- t(apply(credit[,19:24]-credit[,13:18],1,cumsum))
colnames(cum_payment) <- c("CREDIT_DEBT_1","CREDIT_DEBT_2","CREDIT_DEBT_3","CREDIT_DEBT_4","CREDIT_DEBT_5","CREDIT_DEBT_6")
credit <- cbind(credit,cum_payment)

# Examine dataset
head(credit)
summary(credit)
credit$default <- as.factor(credit$default)
library(ggplot2)
qplot(credit$LIMIT_BAL, geom="histogram", xlab="Limit Balance", ylab="")
qplot(as.factor(credit$SEX), xlab="Gender", ylab="")
qplot(as.factor(credit$EDUCATION), xlab="Education", ylab="")
qplot(as.factor(credit$MARRIAGE), xlab="Marriage", ylab="")
qplot(as.factor(credit$AGE), xlab="Age", ylab="")
qplot(log(credit$BILL_AMT1), geom="histogram")


# Balance Samples
# balanced sample yields worse results
# credit_0 <- subset(credit, default == 0)
# credit_1 <- subset(credit, default == 1)
# credit_temp <- credit_0[sample(1:nrow(credit_0), nrow(credit_1)),]
# credit <- rbind(credit_temp, credit_1)

# Split into train and test
set.seed(1)
credit.train.id <- sample(1:nrow(credit), 0.8*nrow(credit))
credit.train <- credit[credit.train.id, ]
credit.test <- credit[-credit.train.id, ]
rownames(credit.train) <- seq(1,dim(credit.train)[1])
rownames(credit.test) <- seq(1,dim(credit.test)[1])


# Predict default for next month

# Prepare data
xfactors_train <- credit.train[,2:24]
xfactors_train <- as.matrix(sapply(xfactors_train, as.numeric))
xfactors_test <- credit.test[,2:24]
xfactors_test <- as.matrix(sapply(xfactors_test, as.numeric))
yfactors_train <- as.matrix(as.numeric(credit.train$default)-1)
yfactors_test <- as.matrix(as.numeric(credit.test$default)-1)


# Lasso (1st Specification)
library(glmnet)
grd <- 10 ^ seq(10,-2,length=100)
lasso.credit <- glmnet(xfactors_train,credit.train$default,alpha=1,lambda=grd,family="binomial")
set.seed(1)
cv.out <- cv.glmnet(xfactors_train,credit.train$default,alpha=1,nfolds=5,family="binomial")
plot(cv.out)
bestlam <- cv.out$lambda.min
out <- glmnet(xfactors_train,credit.train$default,nlambda=1,lambda=bestlam,alpha=1,family="binomial")
coef(out)
lasso.pred <- predict(out,s=bestlam,xfactors_test,type="response")
lasso.pred <- ifelse( lasso.pred[, 1] >= mean(as.numeric(credit.train$default)-1), 1, 0)  # -1 because as.numeric gives 1 and 2 as factors
lasso.table <- table(lasso.pred, credit.test$default)
lasso.accuracy <- (lasso.table[1,1]+lasso.table[2,2])/sum(lasso.table)

# Lasso (2nd Specification)
xfactors2_train <- cbind(as.matrix(credit.train[,2:6]),as.matrix(credit.train[,26:31]))
lasso.credit2 <- glmnet(xfactors2_train,credit.train$default,alpha=1,lambda=grd,family="binomial")
set.seed(1)
cv.out2 <- cv.glmnet(xfactors2_train,credit.train$default,alpha=1,nfolds=5,family="binomial")
plot(cv.out2)
bestlam2 <- cv.out2$lambda.min
out2 <- glmnet(xfactors2_train,credit.train$default,nlambda=1,lambda=bestlam,alpha=1,family="binomial")
coef(out2)
lasso.pred2 <- predict(out,s=bestlam2,xfactors_test,type="response")
lasso.pred2 <- ifelse( lasso.pred2[, 1] >= mean(as.numeric(credit.train$default)-1), 1, 0)  # -1 because as.numeric gives 1 and 2 as factors
lasso.table2 <- table(lasso.pred2, credit.test$default)
lasso.accuracy2 <- (lasso.table2[1,1]+lasso.table2[2,2])/sum(lasso.table2)


# Random Forest (1st Specification)
library(randomForest)
library(MASS)
set.seed(2)
rf.credit <- randomForest(default ~.-ID -default -CREDIT_DEBT_1 -CREDIT_DEBT_2 -CREDIT_DEBT_3 -CREDIT_DEBT_4 -CREDIT_DEBT_5 -CREDIT_DEBT_6, data = credit, subset = credit.train.id, mtry = 5, importance = TRUE)
importance(rf.credit)
rf.pred <- predict(rf.credit, credit.test, type="class")
rf.table <- table(rf.pred, credit.test$default)
rf.accuracy <- (rf.table[1,1]+rf.table[2,2])/sum(rf.table)

# Random Forest (2nd Specification)
rf.credit2 <- randomForest(default ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + CREDIT_DEBT_1 + CREDIT_DEBT_2 + CREDIT_DEBT_3 + CREDIT_DEBT_4 + CREDIT_DEBT_5 + CREDIT_DEBT_6, data = credit, subset = credit.train.id, mtry = 5, importance = TRUE)
importance(rf.credit2)
rf.pred2 <- predict(rf.credit2, credit.test, type="class")
rf.table2 <- table(rf.pred2, credit.test$default)
rf.accuracy2 <- (rf.table2[1,1]+rf.table2[2,2])/sum(rf.table2)


# Neural Network
library(RSNNS)
x_scale_train <- scale(xfactors_train)
x_scale_test <- scale(xfactors_test)
credit.nn1 <- mlp(x_scale_train, yfactors_train, size = c(4), maxit = 10000, learnFuncParams = 0.9, linOut = FALSE)
nn.pred <- predict(credit.nn1, xfactors_test)
nn.pred <- ifelse( nn.pred >= mean(yfactors_train), 1, 0)
nn.table <- table(nn.pred, yfactors_test)
nn.accuracy <- (nn.table[1,1]+nn.table[2,2])/sum(nn.table)

# Ensemble
size_list <- list(c(5),c(6),c(7),c(8),c(5,2),c(5,3),c(5,4),c(5,5),c(6,2),c(6,3),c(6,4),c(6,5),c(6,6),c(7,2),c(7,3),c(7,4),c(7,5),c(7,6),c(7,7),c(8,2),c(8,3),c(8,4),c(8,5),c(8,6),c(8,7),c(8,8))
for (i in 1:15)
{
  credit.nn_temp <- mlp(x_scale_train, yfactors_train, size = size_list[[i]], maxit = 10000, learnFuncParams = 0.01, linOut = FALSE)
  nn.pred <- cbind(nn.pred, predict(credit.nn_temp, x_scale_test))
}

nn_ensemble.pred <- rowMeans(nn.pred)
nn_ensemble.pred <- ifelse( nn_ensemble.pred >= mean(yfactors_train), 1, 0)
nn_ensemble.table <- table(nn_ensemble.pred, yfactors_test)
nn_ensemble.accuracy <- (nn_ensemble.table[1,1]+nn_ensemble.table[2,2])/sum(nn_ensemble.table)
