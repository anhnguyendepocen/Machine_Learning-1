
# Install Tensorflow in R
install.packages("tensorflow")
library(Rcpp)
library(tensorflow)

sess = tf$Session()

# Helloworld
hello <- tf$constant("Hello, TensorFlow!")
sess$run(hello)

# Read in dataset
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

# Placehold an image that has 28*28 = 784 pixels, 9 classes (possible values) for each pixel 
x <- tf$placeholder(tf$float32, shape(NULL, 784L))

# Define the weight matrix
W <- tf$Variable(tf$zeros(shape(784L, 10L)))

# Define the bias matrix
b <- tf$Variable(tf$zeros(shape(10L)))

# Define the output matrix
y <- tf$nn$softmax(tf$matmul(x, W) + b)
y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))

# Define the loss function (cross entropy)
x_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y), reduction_indices=1L))

# Specify the model
optimizer <- tf$train$GradientDescentOptimizer(0.5)
train_step <- optimizer$minimize(x_entropy)
init <- tf$global_variables_initializer()
sess <- tf$Session()
sess$run(init)

# Train the model with 1000 iterations
for (i in 1:1000)
{
  batches <- mnist$train$next_batch(100L)
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  sess$run(train_step, feed_dict = dict(x = batch_xs, y_ = batch_ys))
}

# Make predictions and assess the accuracy
correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(accuracy, feed_dict=dict(x = mnist$test$images, y_ = mnist$test$labels))





