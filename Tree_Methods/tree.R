install.packages("tree")
install.packages("ISLR")

library(tree)
library(ISLR)

# Classification Tree

High <- ifelse(Carseats$Sales <= 8, "No", "Yes")
Carseats <- data.frame(Carseats, High)
tree.carseats <- tree(High ~. -Sales, Carseats)  # use every variables except sales

plot(tree.carseats)
text(tree.carseats, pretty = 0, cex = .5)

set.seed(2)
train <- sample(1:nrow(Carseats), 200)
Carseats.test <- Carseats[-train, ]
High.test <- High[-train]
tree.carseats <- tree(High ~. -Sales, Carseats, subset=train)
tree.pred <- predict(tree.carseats, Carseats.test, type="class")
table(tree.pred, High.test)

set.seed(3)
cv.carseats <- cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)
cv.carseats

par(mfrow = c(1, 2))
plot(cv.carseats$size, cv.carseats$dev,type = "b")
plot(cv.carseats$k, cv.carseats$dev,type = "b")

prune.carseats <- prune.misclass(tree.carseats, best = 9)
plot(prune.carseats)
text(prune.carseats, pretty = 0, cex = 0.5)

tree.pred <- predict(prune.carseats,Carseats.test,type = "class")
table(tree.pred, High.test)


# Random Forests

install.packages("randomForest")
library(randomForest)
library(MASS)

set.seed(1)
train <- sample(1:nrow(Boston), nrow(Boston) / 2)
rf.boston <- randomForest(medv~., data = Boston, subset = train, mtry = 6, importance = TRUE)
importance(rf.boston)

boston.test <- Boston[-train, "medv"]
yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)

plot(yhat.rf, boston.test)
abline(0,1)


# Boosting

install.packages("gbm")
library(gbm)

library(MASS)
head(Boston,2)

set.seed(1)
boost.boston <- gbm(medv ~., data = Boston[train, ], distribution = "gaussian", n.trees = 5000, interaction.depth = 4)

summary(boost.boston)







