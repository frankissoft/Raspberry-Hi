import pandas as pd
import cv2
import numpy as np
import os

csv_path = './data/fer2013/fer2013.csv'
jpg_path = './data/'
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


def load_data(csv_file):
	data = pd.read_csv(csv_file)
	pixels = data['pixels'].tolist()
	width, height = 48,48
	faces = []
	i = 0
	for pixel_sequence in pixels:
		face = [int(pixel) for pixel in pixel_sequence.split(' ')]
		face = np.asarray(face).reshape(width, height)
		faces.append(face)
	faces = np.asarray(faces)
	faces = np.expand_dims(faces, -1)
	emotions = data['emotion'].tolist()

	#emotions = pd.get_dummies(data['emotion']).as_matrix()
	print(faces.shape, len(emotions))
	return faces, emotions

def convert_data(faces, emotions, dest):
	num = 0
	#cv2.imshow('image', faces[0])
	#cv2.waitKey (0)
	for face, emotion in faces, emotions:
		#print(face.shape)
		print(emotion)
		dest = dest + EMOTIONS[emotion]

		cv2.imwrite(dest + str(num) +'.jpg', face)
		print(str(num) + ' OK!')
		num = num + 1





fer2013, fer2013_emotion = load_data(csv_path)
convert_data(fer2013, fer2013_emotion, jpg_path)
print('OK!')
# print(faces.shape)
# print(faces[0])
# print(faces[0].shape)