import os
import sys

import numpy as np
import tensorflow as tf

from utils import *

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def deepnn(x):
  x_image = tf.reshape(x, [-1, 48, 48, 1])

  # conv1
  W_conv1 = weight_variables([5, 5, 1, 48])
  b_conv1 = bias_variable([48])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = maxpool(h_conv1)      #24 * 24 * 48
  # norm1 = tf.nn.lrn(h_pool1, 5, bias=2.0, alpha=2e-5, beta=0.75)

  # conv2
  W_conv2 = weight_variables([3, 3, 48, 128])
  b_conv2 = bias_variable([128])
  h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
  h_pool2 = maxpool(h_conv2)      #12 * 12 * 128
  # norm2 = tf.nn.lrn(h_pool2, 5, bias=2.0, alpha=2e-5, beta=0.75)

  # conv2
  W_conv3 = weight_variables([3, 3, 128, 192])
  b_conv3 = bias_variable([192])
  h_conv3 = conv2d(h_pool2, W_conv3) + b_conv3    #12 * 12 * 196

  # conv4
  W_conv4 = weight_variables([3, 3, 192, 192])
  b_conv4 = bias_variable([192])
  h_conv4 = conv2d(h_conv3, W_conv4) + b_conv4    #12 * 12 * 196

  # conv5
  W_conv5 = weight_variables([3, 3, 192, 128])
  b_conv5 = bias_variable([128])
  h_pool5 = maxpool(h_conv5)    #6 * 6 * 128

  # Fully connected layer
  W_fc1 = weight_variables([6 * 6 * 128, 2048])
  b_fc1 = bias_variable([2048])
  h_pool5_flat = tf.reshape(h_conv5, [-1, 6 * 6 * 128])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
  h_do1 = tf.nn.dropout(h_fc1, 0.5)

  # Fully connected layer
  W_fc2 = weight_variables([2048, 2048])
  b_fc2 = bias_variable([2048])
  h_fc2 = tf.nn.relu(tf.matmul(h_do1, W_fc2) + b_fc2)
  h_do2 = tf.nn.dropout(h_fc2, 0.5)

  # linear
  W_fc3 = weight_variables([2048, 7])
  b_fc3 = bias_variable([7])
  y_conv = tf.add(tf.matmul(h_do2, W_fc3), b_fc3)

  return y_conv


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool(x, k=3, s=2):
  return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                        strides=[1, s, s, 1], padding='SAME')


def weight_variables(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def train_model(train_data):
  fer2013 = input_data(train_data)
  max_train_steps = 30001

  x = tf.placeholder(tf.float32, [None, 2304])
  y_ = tf.placeholder(tf.float32, [None, 7])

  y_conv = deepnn(x)

  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for step in range(max_train_steps):
      batch = fer2013.train.next_batch(50)
      if step % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1]})
        print('step %d, training accuracy %g' % (step, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})

      if step + 1 == max_train_steps:
        saver.save(sess, './models/emotion_model', global_step=step + 1)
      if step % 1000 == 0:
        print('*Test accuracy %g' % accuracy.eval(feed_dict={
          x: fer2013.validation.images, y_: fer2013.validation.labels}))


def predict(image=[[0.1] * 2304]):
  x = tf.placeholder(tf.float32, [None, 2304])
  y_conv = deepnn(x)

  # init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  probs = tf.nn.softmax(y_conv)
  y_ = tf.argmax(probs)

  with tf.Session() as sess:
    # assert os.path.exists('/tmp/models/emotion_model')
    ckpt = tf.train.get_checkpoint_state('./models')
    print(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print('Restore ssss')
    return sess.run(probs, feed_dict={x: image})


def image_to_tensor(image):
  tensor = np.asarray(image).reshape(-1, 2304) * 1 / 255.0
  return tensor


def validate_model(modelPath, validFile):
  x = tf.placeholder(tf.float32, [None, 2304])
  y_conv = deepnn(x)
  probs = tf.nn.softmax(y_conv)

  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(modelPath)

  with tf.Session() as sess:
    print(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print('Restore model sucsses!!')

    files = os.listdir(validFile)

    for file in files:
      if file.endswith('.jpg'):
        image_file = os.path.join(validFile, file)
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        tensor = image_to_tensor(image)
        result = sess.run(probs, feed_dict={x: tensor})
        print(file, EMOTIONS[result.argmax()])