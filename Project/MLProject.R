
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


# Split into train and test
set.seed(1)
credit.train.id <- sample(1:nrow(credit), 0.8*nrow(credit))
credit.train <- credit[credit.train.id, ]
credit.test <- credit[-credit.train.id, ]
rownames(credit.train) <- seq(1,dim(credit.train)[1])
rownames(credit.test) <- seq(1,dim(credit.test)[1])


# Predict default for next month

xfactors_train <- credit.train[,2:24]
xfactors_train <- as.matrix(sapply(xfactors_train, as.numeric))
xfactors_test <- credit.test[,2:24]
xfactors_test <- as.matrix(sapply(xfactors_test, as.numeric))
# OR
xfactors_train <- data.matrix(credit.train[,2:24], rownames.force=NA)
xfactors_test <- data.matrix(credit.test[,2:24], rownames.force=NA)
xfactors_train <- sapply(xfactors_train, as.numeric)
xfactors_test <- sapply(xfactors_test, as.numeric)



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
credit.net1 <- RSNNS::mlp(x=xfactors_train,y=as.numeric(credit.train$default),size = c(10), maxit = 10000, learnFuncParams = 0.0001, linOut = FALSE)

# OR
library(RcppDL)
yfactors_train <- cbind(as.numeric(credit.train$default)-1, as.numeric(ifelse(credit.train$default==1,0,1)))
yfactors_train <- as.matrix(yfactors_train)
hidden <- c(10,3)  # How to select the best size?
credit.deep_auto <- RcppDL::Rsda(xfactors_train, yfactors_train, hidden)
RcppDL::setCorruptionLevel(credit.deep_auto, x = 0.3)
RcppDL::setFinetuneLearningRate(credit.deep_auto, x = 0.1)
summary(credit.deep_auto)
pretrain(credit.deep_auto)
finetune(credit.deep_auto)
deep_auto.pred <- predict(credit.deep_auto, xfactors_test)
deep_auto.pred <- ifelse( deep_auto.pred[, 1] >= mean(yfactors_train[,1]), 1, 0)
deep_auto.table <- table(deep_auto.pred, yfactors_test[,1])
deep_auto.accuracy <- (deep_auto.table[1,1]+deep_auto.table[2,2])/sum(deep_auto.table)



# Stacked Autoencoder
library(autoencoder)
library(SAENET)
credit.ae <- SAENET.train(as.matrix(credit.train[,2:24]), n.nodes = c(5,3), lambda = 1e-5, beta = 1e-5, rho = 0.01, epsilon = 0.01)
ae.pred <- SAENET.predict(credit.ae, as.matrix(credit.test[,2:24]),layers=c(1),all.layers=FALSE)
# How to use SAE for supervised learning?



# Ensemble

























# Backup






# Simple classification tree
library(tree)

# Gini
tree.credit.gini <- tree(default ~., data = credit, subset = credit.train.id, control = tree.control(nrow(credit.train)/2, mincut = 5), split = "gini")
tree.pred.gini <- predict(tree.credit.gini, credit.test, type="class")
gini.table <- table(tree.pred.gini, credit.test$y)
gini.accuracy <- (gini.table[1,1]+gini.table[2,2])/sum(gini.table)
plot(tree.credit.gini)
text(tree.credit.gini, pretty = 0, cex = .5)

# Deviance
tree.credit.deviance <- tree(default ~., data = credit, subset = credit.train.id, control = tree.control(nrow(credit.train)/2, mincut = 5), split = "deviance")
tree.pred.deviance <- predict(tree.credit.deviance, credit.test, type="class")
deviance.table <- table(tree.pred.deviance, credit.test$y)
deviance.accuracy <- (deviance.table[1,1]+deviance.table[2,2])/sum(deviance.table)
plot(tree.credit.deviance)
text(tree.credit.deviance, pretty = 0, cex = .5)


# Boosted Trees
library(adabag)
boost.credit <- boosting(default ~ AGE + MARRIAGE, data = credit.train, boos = TRUE, control = rpart.control(minsplit = 0))
boost.credit$importance
boost.pred <- predict(boost.credit, credit.test)
boost.accuracy <- (boost.pred$confusion[1,1]+boost.pred$confusion[2,2])/sum(boost.pred$confusion)
