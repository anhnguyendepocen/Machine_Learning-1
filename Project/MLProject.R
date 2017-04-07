
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

# Split into train and test
set.seed(1)
credit.train.id <- sample(1:nrow(credit), 0.8*nrow(credit))
credit.train <- credit[credit.train.id, ]
credit.test <- credit[-credit.train.id, ]
rownames(credit.train) <- seq(1,dim(credit.train)[1])
rownames(credit.test) <- seq(1,dim(credit.test)[1])


# Predict default for next month


# Random Forest (1st Specification)
library(randomForest)
library(MASS)
set.seed(2)
rf.credit <- randomForest(default ~.-ID -default -CREDIT_DEBT_1 -CREDIT_DEBT_2 -CREDIT_DEBT_3 -CREDIT_DEBT_4 -CREDIT_DEBT_5 -CREDIT_DEBT_6, data = credit, subset = credit.train.id, mtry = 5, importance = TRUE)
importance(rf.credit)
rf.pred <- predict(rf.credit, credit.test, type="class")
rf.pred[rf.pred < 0.5] <- 0
rf.pred[rf.pred > 0.5 | rf.pred == 0.5] <- 1
rf.table <- table(rf.pred, credit.test$default)
rf.accuracy <- (rf.table[1,1]+rf.table[2,2])/sum(rf.table)

# Random Forest (2nd Specification)
rf.credit2 <- randomForest(default ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + CREDIT_DEBT_1 + CREDIT_DEBT_2 + CREDIT_DEBT_3 + CREDIT_DEBT_4 + CREDIT_DEBT_5 + CREDIT_DEBT_6, data = credit, subset = credit.train.id, mtry = 5, importance = TRUE)
importance(rf.credit2)
rf.pred2 <- predict(rf.credit2, credit.test, type="class")
rf.pred2[rf.pred2 < 0.5] <- 0
rf.pred2[rf.pred2 > 0.5 | rf.pred2 == 0.5] <- 1
rf.table2 <- table(rf.pred2, credit.test$default)
rf.accuracy2 <- (rf.table2[1,1]+rf.table2[2,2])/sum(rf.table2)



# Lasso
install.packages("glmnet")
library(glmnet)
grd <- 10 ^ seq(10,-2,length=100)
xfactors <- credit.train[,2:23]
xfactors <- as.matrix(xfactors)
lasso.credit <- glmnet(xfactors,credit.train$default,alpha=1,lambda=grd,family="binomial")
set.seed(1)
cv.out <- cv.glmnet(xfactors,credit.train$default,alpha=1,nfolds=5,family="binomial")
plot(cv.out)
bestlam <- cv.out$lambda.min
out <- glmnet(xfactors,credit.train$default,nlambda=1,lambda=bestlam,alpha=1,family="binomial")
coef(out)



# Neural Network
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
