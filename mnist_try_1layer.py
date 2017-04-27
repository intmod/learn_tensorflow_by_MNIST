# https://www.tensorflow.org/get_started/mnist/beginners
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("MNIST data ready for analysis!\n")  # get data ready

# model
x = tf.placeholder(tf.float32, [None, 784])  # placeholder for inputing imgs
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)  # model defined here!
y_ = tf.placeholder(tf.float32, [None, 10])  # will be loaded in sess.run()

# loss
# cross entropy
# --- note that, y_ and y are not multiplied as matrices
#     instead, merely the corresponding elements are multiplied
#     so the dim of `y_ * tf.log(y)` is [batch_size, 10]
#     reduce_sum works on the 1-th dim (not 0-th), so the result dim is batch_size
#     lastly, reduce_mean works on the batch to get a scalar
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# It would be very similar for the Euclidean distance case, as is shown below:
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices=[1]))
train_stepper = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# initializer
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()  # FIXED for new version tensorflow

# training
batch_size = 100  # None used in the above placeholder will be this number
train_step_number = 1000
for i in range(train_step_number):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size) # load in each step
    sess.run(train_stepper, feed_dict={x: batch_xs, y_: batch_ys})
    if (i % 100) == 0: print('\n', i,':\n', sess.run(W), '\n', sess.run(b), '\n')

# evaluation
# arg_max : the entry with the highest probability is our prediction
if_prediction_correct = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))  # T,F,T,...
accuracy = tf.reduce_mean(tf.cast(if_prediction_correct, "float"))     # 1,0,1,...
print("Accuarcy on Test-dataset: ",  \
      sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
