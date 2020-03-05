import collections

import numpy as np
import pandas as pd
import cv2
from tensorflow.python.framework import dtypes, random_seed


def load_data(data_file):
	"""Fetch all the data from a csv file.

	Retrive pixels and emotion labels from csv file. One row stores all the
	labels using numbers 0-6, the other row stores all the pixel values. Each
	cell stores all the pixel values of one human facial expression image,
	which has 2304(48*48) pixels. The values are integer between 0 and 255.
	The values in each cell are seperated by spaces.

	Args:
		data_file (str):A csv file containing pixel values and labels

	Returns:
		faces (np.ndarray):A 4-D array representing the pixel values of all the data
			[NUMBER OF IMAGES, 48, 48, 1]
		emotions (np.ndarray):A 1-D array of integers(0~6) representing the labels of all 
			the corresponding images.

	"""
	data = pd.read_csv(data_file)

	pixels = data['pixels'].tolist()
	width, height = 48, 48
	#Set the resolution of the images

	face = []
	for pixel_seq in pixels:
		face = [int(pixel) for pixel in pixel_seq.split(' ')]
		face = np.asarray(face).reshape(width, height)
		faces.append(face)

	faces = np.asarray(faces)
	faces = np.expand_dims(faces, -1)
	# Add one dimension so that the array is now 3-D
	emotions = pd.get_dummies(data['emotion']).as_matrix()
	# Use one-hot encoding to get normalized labels for a better understanding
	# For example:
	"""
	emotion | happy | sad | angry | fear | disgusted | surprised | neutral |
	image1  | 1	    | 0   | 0     | 0    | 0         | 0         | 0       |
	image2  | 0	    | 0   | 0     | 1    | 0         | 0         | 0       |
	...     | ...   | ... | ...   | ...  | ...       | ...       | ...     |
	
	"""
	return faces, emotions

class DataSet(object):
	"""Manipulate the dataset object for training, validation and test in ML.
	
	Store all the data from the object in Dataset class. Wrap images in small batches, shuffle 
	in each batch and train each at a time. Use 1-D array of float number(0~1) as pixel values 
	to represent an image. Split the dataset into three parts: training, validation and test.

	Attributes:
		_num_examples (int): Number of examples(images) from the object.
		_index_in_epoch (int): Index of the current epoch.
		_epochs_completed (int): Number of epochs that has been used in training process.
		_images (numpy.ndarray): Array of pixel values of images.
		_labels (): One-hot encoding of facial expression features.

	"""
	def __init__(self,
				 images,
				 labels,
				 reshape=True,
				 dtype=dtypes.float32,
				 seed=None):
		"""Inits DataSet with data inputs and manipulation parameters

		Args:
			images (numpy.ndarray): Array of pixel values of images
			labels (numpy.ndarray): One-hot encoding of facial expression features
			reshape (binary): Whether to shuffle the data
			dtype (tf.dtypes.DType): Data type of images and labels
			seed (int):Determine wheteher to use the same random numbers.
		
		Returns: None
		"""
		# Returns two seed derived from graph-level and op-level seeds
		seed1, seed2 = random_seed.get_seed(seed)
		
		# Choose the proper seed between the two seed
		np.random.seed(seed1 if seed is None else seed2)
		
		# Reshape so that each image becomes an 1-D array
		if reshape:
			assert images.shape[3] == 1
			images = images.reshape(images.shape[0],
									images.shape[1] * images.shape[2])

		# Rescale values of pixel to 0.0~1.0 for a prompt computation
		if dtype == dtypes.float32:
			images = images.astype(np.float32)
			images = np.multiply(images, 1.0 / 255.0)


		self._num_examples = images.shape[0]
		self._index_in_epoch = 0
		self._epochs_completed = 0
		self._images = images
		self._labels = labels

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels
	
	@property
	def num_examples(self):
		return self._num_examples
	
	@property
	def epochs_completed(self):
		return self._epochs_completed
	
	def next_batch(self,
				   batch_size,
				   shuffle=True):
		"""Shuffle the images in small batches
		
		Fetch images and wrap them into small batches for training process.
		Get the next batch from remaining images

		Args:
			batch_size (int): Numbers of images in each batch
			shuffle (bool): Whether to shuffle the images in each batch

		Returns:
			"batch1, batch2" (tuple): A tuple consists of an image batch and a label batch

		"""
		start = self._index_in_epoch
		# First batch
		if self._epochs_completed == 0 and start == 0 and shuffle:
			perm0 = np.arange(self._num_examples)
			np.random.shuffle(perm0)
			self._images = self._images[perm0]
			self._labels = self._labels[perm0]
		
		# The last batch(size < batch_size)
		if start + batch_size > self._num_examples:
			self._epochs_completed += 1
			rest_num_examples = self._num_examples - start
			images_rest_part = self._images[start : self._num_examples]
			labels_rest_part = self._labels[start : self._num_examples]
			
			# Shuffle
			if shuffle:
				perm = np.arange(self._num_examples)
				np.random.shuffle(perm)
				self._images = self._images[perm]
				self._labels = self._labels[perm]
			
			start = 0
			self._index_in_epoch = batch_size - rest_num_examples
			end = self._index_in_epoch
			images_new_part = self._images[start:end]
			labels_new_part = self._labels[start:end]
			return (np.concatenate((images_rest_part, images_new_part), axis=0), 
					np.concatenate((labels_rest_part, labels_new_part), axis=0))
		
		# Next batch(start from the end of last batch)
		else:
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
			return self._images[start : end], self._labels[start : end]

def input_data(training_dir,
			   dtype=dtypes.float32,
			   reshape=True,
			   seed=None):
	"""Get the data and split them into sets.
	
	Split the data into training set, validation set and test set according to the 
	pre-defined size of each set.

	Args:
		train_dir (str): The directory that stores the data file.
		dtype (tf.dtypes.DType): Data type of input data.
		reshape :(bool): Whether to reshape the input data.
		seed (int):Determine wheteher to use the same random numbers.
	
	"""
	training_set_size = 28709        #80% of all the data
	validation_set_size = 3589        #10% of all the data
	test_set_size = 3589        #10% of all the data

	training_faces, training_emotions = load_data(training_dir)
	print('Dataset input OK!')
	
	# Validate
	validation_faces = training_faces[training_set_size : training_set_size + validation_set_size]
	validation_emotions = training_emotions[training_set_size : training_set_size + validation_set_size]
	
	# Test
	test_faces = training_faces[training_set_size + validation_set_size : ]
	test_emotions = training_emotions[training_set_size + validation_size : ]
	
	# Train
	training_faces = training_faces[ : training_set_size]
	training_emotions = training_emotions[ : training_set_size]

	# Create a new tuple subclass looked like "Datasets(training, validation, test)"
	Datasets = collections.namedtuple('Datasets', ['training', 'validation', 'test'])
	training = Datasets(training_faces, 
						training_emotions, 
						dtype=dtype, 
						reshape=reshape, 
						seed = seed)
	validation = Dataset(validation_faces, 
						 validation_emotions,
						 dtype=dtype, 
						 reshape=reshape, 
						 seed=seed)
	test = DataSet(test_faces, 
				   test_emotions, 
				   dtype=dtype,
				   reshape=reshape,
				   seed=seed)
	return Datasets(training=training, validation=validation, test=test)

	def _test():
		import cv2
		i = input_data('./data/fer2013/fer2013.csv')

	if __name__ == '__main__':
		_test()