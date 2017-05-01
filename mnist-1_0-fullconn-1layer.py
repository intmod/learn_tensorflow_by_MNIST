#  !/usr/bin/env python
#  -*- coding:utf-8 -*-


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# https://www.tensorflow.org/get_started/mnist/beginners
# https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/
# wmji@github.com


import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)  # tf&DL_without_a_phD


# data
mnist = read_data_sets("MNIST_data/", one_hot=True)
print("MNIST data ready for analysis!\n")  # get data ready
batch_size = 100  # how many imgs in each batch?


# model
# neural network with 1 layer of 10 softmax neurons  - tf&DL_without_a_phD
#
# · · · · · · · · · ·       (input data, flattened pixels)   x [batch, 784]
# \*/*\*/*\*/*\*/*\*/    -- fully connected layer (softmax)  W [784, 10]   b[10]
#   · · · · · · · ·                                          y [batch, 10]
#
x = tf.placeholder(tf.float32, [None, 784]) # for inputing imgs, be of batchSize
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)  # model defined here! add b to each line
# softmax(matrix) applies softmax on each line
# softmax(line) exp each value then divides by the norm of the resulting line
y_ = tf.placeholder(tf.float32, [None, 10])  # will be loaded in sess.run()


# loss
# cross-entropy = - sum( Y_i * log(Yi) )
#            Y: the computed output vector
#            Y_: the desired output vector
# --- note that, y_ and y are not multiplied as matrices
#     instead, merely the corresponding elements are multiplied
#     so the dim of `y_ * tf.log(y)` is [batch_size, 10]
#     reduce_sum works on the 1-th dim (not 0-th), so resulting dim is batchSize
#     lastly, reduce_mean works on the batch to get a scalar
#loss = -tf.reduce_mean(tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))*100
#       normalized for batches of 100 imgs, to obtain cross-entropy of each img
#     only a reduce_mean would also generate the scalar
loss = -tf.reduce_mean(y_ * tf.log(y))*1000  # *100*10
#      *10 because  "mean" included an unwanted division by 10
# It would be very similar for the Euclidean distance case, as is shown below:
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices=[1]))
train_stepper = tf.train.GradientDescentOptimizer(0.001).minimize(loss)


# initializer
init = tf.global_variables_initializer()  # note the version problem


# evaluation
# arg_max : the entry with the highest probability is our prediction
if_prediction_correct = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1)) # T,F,T...
accuracy = tf.reduce_mean(tf.cast(if_prediction_correct, tf.float32)) # 1,0,1...


with tf.Session() as sess:
    sess.run(init)

    # training
    for i in range(10000):  # train_step_number
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # load
        sess.run(train_stepper, feed_dict={x: batch_xs, y_: batch_ys})
        if (i % 1000) == 0: print(i,':\n', sess.run(W), '\n', sess.run(b), '\n')

    print("Accuarcy on Test-dataset: ",  \
      sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))


print("\nDone.")  # all done
