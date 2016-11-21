import tensorflow as tf
import numpy as np

# The variables which we want to find out. For this demo, we want to find out the equation of the line in form of y = mx + b
W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))

# The data
x = tf.constant([-1.0, 0.0, 1.0])
y = tf.constant([-2.0, 0.0, 2.0])

# Our hypothesis
y_ = tf.mul(x,W) + b

# The loss function
loss_func = tf.reduce_sum(tf.square(y - y_))

# Running the graph
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss_func, var_list=[W,b])
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for _ in xrange(100):
		_,w,_ = sess.run([train_step,W,b])
		print w