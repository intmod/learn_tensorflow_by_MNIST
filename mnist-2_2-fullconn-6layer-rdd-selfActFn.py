#  !/usr/bin/env python
#  -*- coding:utf-8 -*-


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# https://www.tensorflow.org/get_started/mnist/beginners
# https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/
# http://stackoverflow.com/questions/39921607/\
#   tensorflow-how-to-make-a-custom-activation-function-with-only-python
# mw58@github.com

# this codes shows how to use user-defined activation function
# define a good activation function
# or, accept a bad result, as shown in this code.

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.framework import ops
tf.set_random_seed(0)  # tf&DL_without_a_phD


# self-def activation function:
def triple(x): return x**3/3.0  # this is the activation function
np_triple = np.vectorize(triple)   # make it a numpy function

def d_triple(x): return 1.0* x**2  # gradient of the activation
np_d_triple = np.vectorize(d_triple)

### make a numpy function to a tensorflow function
### tf.py_func(func, inp, Tout, stateful=stateful, name=name) do transformation
###    https://www.tensorflow.org/api_docs/python/tf/py_func
np_d_triple_32 = lambda x: np_d_triple(x).astype(np.float32)

def tf_d_triple(x,name=None):
    with ops.name_scope(name, "d_triple", [x]) as name:
        y = tf.py_func(np_d_triple_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]

### hack to get gradients
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+10))
    # ...
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def triple_grad(op, grad):
    x = op.inputs[0]
    # ...
    n_gr = tf_d_triple(x)
    return grad * n_gr

### combining it all together
np_triple_32 = lambda x: np_triple(x).astype(np.float32)
# ...
def tf_triple(x, name=None):
    # ...
    with ops.name_scope(name, "triple", [x]) as name:
        y = py_func(np_triple_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=triple_grad)  # <-- here's the call to the gradient
        return y[0]


# data
mnist = read_data_sets("../MNIST-data/", one_hot=True)
print("MNIST data ready for analysis!\n")  # get data ready
batch_size = 100  # how many imgs in each batch?


# model
# neural network with 6 layers of 400,200,100,50,25,10 neurons, respectively
# this is similar with mnist_1layer.py, only that a few sigmoid layers added
# UPDATE:
# 1 use self-defined transfer function, instead of sigmoid/relu
nn0,nn1,nn2,nn3,nn4,nn5,nn6 = 784,400,200,100,50,25,10  # neuroNumber each layer
x = tf.placeholder(tf.float32, [None, nn0]) # for inputing imgs, be of batchSize
transFunc = tf_triple # tf.nn.relu  # tf.nn.sigmoid
# Probability of keeping a node during dropout
#  =1 at test time (no dropout) and ~0.7 at training time.
pkeep = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([nn0, nn1], stddev=0.1)) #784=28*28, inputs
b1 = tf.Variable(tf.zeros([nn1])+0.01)
y1 = transFunc(tf.matmul(x, W1) + b1)
y1d = tf.nn.dropout(y1, pkeep)

W2 = tf.Variable(tf.truncated_normal([nn1, nn2], stddev=0.1))
b2 = tf.Variable(tf.zeros([nn2])+0.01)
y2 = transFunc(tf.matmul(y1d, W2) + b2)
y2d = tf.nn.dropout(y2, pkeep)

W3 = tf.Variable(tf.truncated_normal([nn2, nn3], stddev=0.1))
b3 = tf.Variable(tf.zeros([nn3])+0.01)
y3 = transFunc(tf.matmul(y2d, W3) + b3)
y3d = tf.nn.dropout(y3, pkeep)

W4 = tf.Variable(tf.truncated_normal([nn3, nn4], stddev=0.1))
b4 = tf.Variable(tf.zeros([nn4])+0.01)
y4 = transFunc(tf.matmul(y3d, W4) + b4)
y4d = tf.nn.dropout(y4, pkeep)

W5 = tf.Variable(tf.truncated_normal([nn4, nn5], stddev=0.1))
b5 = tf.Variable(tf.zeros([nn5])+0.01)
y5 = transFunc(tf.matmul(y4d, W5) + b5)
y5d = tf.nn.dropout(y5, pkeep)

W6 = tf.Variable(tf.truncated_normal([nn5, nn6], stddev=0.1))
b6 = tf.Variable(tf.zeros([nn6])+0.01)
yLogits = tf.matmul(y5d, W6) + b6
y = tf.nn.softmax(yLogits)

y_ = tf.placeholder(tf.float32, [None, nn6])  # will be loaded in sess.run()


# loss
# cross-entropy loss function (= -sum(Y_i * log(Yi)) )
#    normalised for batches of 100 images
# use softmax_cross_entropy_with_logits to avoid numerical stability problems,
# ie, log(0) is NaN
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
                      (logits=yLogits, labels=y_))*100


# training
# learning rate decay
max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
# variable learning rate
lr = tf.placeholder(tf.float32)
# the optimizer
train_stepper = tf.train.AdamOptimizer(lr).minimize(loss)


# initializer
init = tf.global_variables_initializer()  # note the version problem


# evaluation
# arg_max : the entry with the highest probability is our prediction
if_prediction_correct = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1)) # T,F,T...
accuracy = tf.reduce_mean(tf.cast(if_prediction_correct, tf.float32)) # 1,0,1...


with tf.Session() as sess:
    sess.run(init)

    # training
    for i in range(2000):  # train_step_number
        # learning rate decay
        lerate = min_learning_rate + \
                (max_learning_rate-min_learning_rate) * math.exp(-i/decay_speed)
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # load & train:
        sess.run(train_stepper, {x:batch_xs, y_:batch_ys, pkeep:0.7, lr:lerate})
        if i % 100 == 0:
            train_accuacy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, pkeep: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuacy))

    # accuacy on test
    print("test accuracy %g"%(accuracy.eval(\
           feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
    print("Accuarcy on Test-dataset: ", sess.run(accuracy, \
        feed_dict={x:mnist.test.images, y_:mnist.test.labels, pkeep:1}))


print("\nDone.")  # all done
