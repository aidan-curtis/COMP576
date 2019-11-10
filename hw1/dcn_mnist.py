__author__ = 'tan_nguyen'

import os
import time

# Load MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Import Tensorflow and start a session
import tensorflow as tf
sess = tf.InteractiveSession()

def weight_variable(shape):
	'''
	Initialize weights
	:param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
	w: width of the filters
	h: height of the filters
	Cin: the number of the channels of the filters
	Cout: the number of filters
	:return: a tensor variable for weights with initial values
	'''
	W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
	return W

def bias_variable(shape):
	'''
	Initialize biases
	:param shape: shape of biases, e.g. [Cout] where
	Cout: the number of filters
	:return: a tensor variable for biases with initial values
	'''

	b = tf.Variable(tf.constant(0.1, shape=shape))
	return b

def conv2d(x, W):
	'''
	Perform 2-D convolution
	:param x: input tensor of size [N, W, H, Cin] where
	N: the number of images
	W: width of images
	H: height of images
	Cin: the number of channels of images
	:param W: weight tensor [w, h, Cin, Cout]
	w: width of the filters
	h: height of the filters
	Cin: the number of the channels of the filters =
	the number of channels of images
	Cout: the number of filters
	:return: a tensor of features extracted by the
	filters, a.k.a. the results after convolution
	'''
	
	# IMPLEMENT YOUR CONV2D HERE
	
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	'''
	Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
	:param x: input data
	:return: the results of maxpooling (max-marginalized + downsampling)
	'''
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main():
		
	result_dir = './results/'
	max_step = 5500
	
	start_time = time.time() # start timing
	
	# placeholders for input data and input labeles
	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10])
	
	# reshape the input image
	x_image = tf.reshape(x, [-1, 28, 28, 1])
	
	# first convolutional layer
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	
	tf.summary.histogram("Layer_1_Input", x_image)
	tf.summary.histogram("Layer_1_Weights", W_conv1)
	tf.summary.histogram("Layer_1_Bias", b_conv1)
	tf.summary.histogram("Layer_1_Activations", h_conv1)
	tf.summary.histogram("Layer_1_Pooling", h_pool1)
	
	# second convolutional layer
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	
	tf.summary.histogram("Layer_2_Input", h_pool1)
	tf.summary.histogram("Layer_2_Weights", W_conv2)
	tf.summary.histogram("Layer_2_Bias", b_conv2)
	tf.summary.histogram("Layer_2_Activations", h_conv2)
	tf.summary.histogram("Layer_2_Pooling", h_pool2)
	
	# densely connected layer
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	tf.summary.histogram("fc1_Inputs", h_pool2)
	tf.summary.histogram("fc1_Weights", W_fc1)
	tf.summary.histogram("fc1_Biases", b_fc1)
	tf.summary.histogram("fc1_Activations", h_fc1)
	tf.summary.histogram("fc1_Pool", h_pool2_flat)
	
	# dropout
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	# softmax
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')
	
	tf.summary.histogram("h_fc1_drop", h_pool2)
	tf.summary.histogram("fc2_Weights", W_fc2)
	tf.summary.histogram("fc2_Biases", b_fc2)
	tf.summary.histogram("fc2_Activations", y_conv)
	
	# setup training
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='accuracy')
																								
	# Add a scalar summary for the snapshot loss.
	tf.summary.scalar(cross_entropy.op.name, cross_entropy)
																									
	# Build the summary operation based on the TF collection of Summaries.
	summary_op = tf.summary.merge_all()
	
	validation_acc_summary = tf.summary.scalar('validation_accuracy', accuracy)
	test_acc_summary = tf.summary.scalar('test_accuracy', accuracy)
	
	# Add the variable initializer Op.
	init = tf.initialize_all_variables()
	
	# Create a saver for writing training checkpoints.
	saver = tf.train.Saver()
	
	# Instantiate a SummaryWriter to output summaries and the Graph.
	summary_writer = tf.summary.FileWriter(result_dir, sess.graph)
	
	# Run the Op to initialize the variables.
	sess.run(init)
	
	# run the training
	for i in range(200):
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			# output the training accuracy every 100 iterations
			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
																						 
			summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
			summary_writer.add_summary(summary_str, i)
			summary_writer.flush()
		# save the checkpoints every 1100 iterations
		if i % 1100 == 0 or i == max_step:
			checkpoint_file = os.path.join(result_dir, 'checkpoint')
			saver.save(sess, checkpoint_file, global_step=i)
			test_accuracy = sess.run(test_acc_summary, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
			validation_accuracy = sess.run(validation_acc_summary, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
			summary_writer.add_summary(test_accuracy, i)
			summary_writer.add_summary(validation_accuracy, i)
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



	# print/log test error
	print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images,  y_: mnist.test.labels, keep_prob: 1.0}))

	print(y_conv.eval(feed_dict={x: mnist.test.images}, session = sess))
	stop_time = time.time()
	print('The training takes %f second to finish'%(stop_time - start_time))

if __name__ == "__main__":
	main()
