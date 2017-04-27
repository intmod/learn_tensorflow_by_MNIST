import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# data preprocessing
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("MNIST data ready for analysis!")  # get data ready
x = tf.placeholder(tf.float32, [None, 784])  # placeholder for inputing imgs

# paras
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss func
y_ = tf.placeholder(tf.float32, [None, 10])  # will be loaded in sess.run()
# loss = -tf.reduce_sum(y_ * tf.log(y))  # cross func
loss = (y_ - y) ** 2  # Euc dist sqr
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

# init
init = tf.global_variables_initializer()  # FIXED for new version tensorflow
sess = tf.Session()
sess.run(init)

# train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if (i % 100) == 0: print('\n', i,':\n', sess.run(W), '\n', sess.run(b), '\n')

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Accuarcy on Test-dataset: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
