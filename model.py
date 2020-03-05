import os
import sys

import numpy as np
import tensorflow as tf

from utils import *

# 7 emotion catagories that all the images will fall into
EMOTIONS = ['happy', 'sad', 'fearful', 'disgusted', 'angry', 'surprised', 'neutral']

def conv2d(x, W):
	"""Use conv2d() in tensorflow to compute convolution on x with

	Args:
		x: A tensor containing all the pixel values
		W: A 4-D tensor containing the size and step size of the filter

	Return:
		A tensor of same data type as x

	"""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool(x):

	# Return a tensor of same type as x
	# With a kernel of 3*3 and step of 2*2
	return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variables(shape):
	
	# Store weights as 
	init = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init)


def bias_variables(shape):
	"""
	Args:
		shape (tuple): shape of bias tensor
	Return:
	"""
	init = tf.constant(0.1, shape=shape)
	return tf.Variable(init)


def deepnn(x):
	"""
	Args:
		x (tensor): all the training data samples
	Return:
	"""
	x_image = tf.reshape(x, [-1, 48, 48, 1])

	# Conv layer 1
	W_conv1 = weight_variables([5, 5, 1, 64])
	b_conv1 = bias_variables([64])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    # output size 48*48*64

	# Pooling layer 1
	h_pool1 = maxpool(h_conv1)    # output size 24*24*64

	# Norm layer 1
	norm = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

	# Conv layer 2
	W_conv2 = weight_variables([3, 3, 64, 64])
	b_conv2 = bias_variables([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)    # output size 24*24*64
	
	norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.01/9.0, beta=0.75)
	
	h_pool2 = maxpool(norm2)    # output size 12*12*64

	# Fully connected layer 1
	W_fc1 = weight_variables([12 * 12 * 64, 384])
	b_fc1 = bias_variables([384])
	h_conv3_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

	# Fully connected layer 2
	W_fc2 = weight_variables([384, 192])
	b_fc2 = bias_variables([192])
	h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

	# Linear layer
	W_fc3 = weight_variables([192, 7])
	b_fc3 = bias_variables([7])
	y_conv = tf.add(tf.matmul(h_fc2, W_fc3), b_fc3)

	return y_conv

def train_model(training_data):
	"""
	Args:
		training_data ():
	Return:

	"""
	fer2013 = input_data(training_data)
	max_train_steps = 30001

	x = tf.placeholder(tf.float32, [None, 2304])
	y_ = tf.placeholder(tf.float32, [None, 7])

	y_conv = dnn(x)

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		for step in range(max_train_steps):
			batch = fer2013.train.next_batch(50)

			# Take a look at training accuracy every 100 steps
			if step % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
			print('step %d, training accuracy %g' % (step, train_accuracy))
		
		# Loss
		train_step.run(feed_dict={x: batch[0], y_:batch[1]})

		if step + 1 == max_train_steps:
			saver.save(sess, './models/emotion_model', global_step=step + 1)

		# Take a loog at validation accuracy every 1000 steps
		if step % 1000 == 0:
			print('Test accuracy %g' % accuracy.eval(feed_dict={x: fer2013.validation.images, 
																y: fer2013.validation.labels}))
# Not using
def predict(image=[[0.1] * 2304]):
	x = tf.placeholder(tf.float32, [None, 2304])
	y_conv = deepnn(x)

	saver = tf.train.Saver()
	probs = tf.nn.softmax(y_conv)
	y_ = tf.argmax(probs)

	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state('./models')
		print(ckpt.model_checkpoint_path)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print('Session restored!')
		return sess.run(probs, feed_dict={x: image})


def image_to_tensor(image):

	# Rescale values of pixel to 0.0~1.0 for a prompt computation
	tensor = np.asarray(image).reshape(-1, 2304) * 1 / 255.0
	return tensor


def validate_model(model_path, validation_file):
	"""
	Args:
		model_path (str):
		validataion_file (str):
	
	Return: None

	"""
	x = tf.placeholder(tf.float32, [None, 2304])
	y_conv = dnn(x)
	probs = tf.nn.softmax(y_conv)

	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state(model_path)

	with tf.Session() as sess:
		print(ckpt.model_checkpoint_path)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print('Model restored!')

		files = os.listdir(validation_file)

	for file in files:
		if file.endswith('.jpg'):
			image_file = os.path.join(validation_file, file)
			image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
			tensor = image_to_tensor(image)
			result = sess.run(probs, feed_dict={x: tensor})
			print(file, EMOTIONS[result.argmax()])

