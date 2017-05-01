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
# neural network with 6 layers of 400,200,100,50,25,10 neurons, respectively
# this is similar with mnist_1layer.py, only that a few sigmoid layers added
#
nn0,nn1,nn2,nn3,nn4,nn5,nn6 = 784,400,200,100,50,25,10  # neuronNumber each layer
x = tf.placeholder(tf.float32, [None, nn0]) # for inputing imgs, be of batchSize

W1 = tf.Variable(tf.truncated_normal([nn0, nn1], stddev=0.1)) #784=28*28, inputs
b1 = tf.Variable(tf.zeros([nn1])+0.1)
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([nn1, nn2], stddev=0.1))
b2 = tf.Variable(tf.zeros([nn2])+0.1)
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([nn2, nn3], stddev=0.1))
b3 = tf.Variable(tf.zeros([nn3])+0.1)
y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)

W4 = tf.Variable(tf.truncated_normal([nn3, nn4], stddev=0.1))
b4 = tf.Variable(tf.zeros([nn4])+0.1)
y4 = tf.nn.sigmoid(tf.matmul(y3, W4) + b4)

W5 = tf.Variable(tf.truncated_normal([nn4, nn5], stddev=0.1))
b5 = tf.Variable(tf.zeros([nn5])+0.1)
y5 = tf.nn.sigmoid(tf.matmul(y4, W5) + b5)

W6 = tf.Variable(tf.truncated_normal([nn5, nn6], stddev=0.1))
b6 = tf.Variable(tf.zeros([nn6])+0.1)
yLogits = tf.matmul(y5, W6) + b6
y = tf.nn.softmax(yLogits)

y_ = tf.placeholder(tf.float32, [None, nn6])  # will be loaded in sess.run()


# loss
# cross-entropy loss function (= -sum(Y_i * log(Yi)) )
#    normalised for batches of 100 images
# use softmax_cross_entropy_with_logits to avoid numerical stability problems,
# ie, log(0) is NaN
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
                      (logits=yLogits, labels=y_))*100
train_stepper = tf.train.AdamOptimizer(0.001).minimize(loss)
# well, this optimizer is very important!

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
        if (i % 100) == 0: print(i)

    print("Accuarcy on Test-dataset: ",  \
      sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))


print("\nDone.")  # all done
