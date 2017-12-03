import csv
import cv2
import numpy as np
import sklearn
import tensorflow as tf

# import the necessary Keras packages
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam

import keras
#print(keras.__version__)

DIR_DATA = "./data/"
DRIVING_LOG = './data/driving_log.csv'

# network and training
NB_EPOCH = 8#3#8#6#4# 2#10# 4 #7#20
BATCH_SIZE = 128
VERBOSE = 1
VALIDATION_SPLIT=0.2

INPUT_SHAPE = (160, 320, 3)	

def get_logs(file_path):
	"""get the lines from the given file path as csv file"""
	lines = []
	with open(file_path) as csvfile:
		reader = csv.reader(csvfile)
		fieldnames = next(reader)
		for line in reader:			
			lines.append(line)
	return lines

def equalize_histogram(image):
	"""apply histgram equalization to the given image"""
	copied_image = image.copy()
	copied_image[:,:,0] = cv2.equalizeHist(image[:,:,0])
	copied_image[:,:,1] = cv2.equalizeHist(image[:,:,1])
	copied_image[:,:,2] = cv2.equalizeHist(image[:,:,2])
	return copied_image

def process_image(image):
	"""process an image"""
	processed_image = equalize_histogram(image)
	return processed_image


def get_data_pair(rows):
	"""load data set from training logs"""
	car_images = []
	steering_angles = []
	for row in rows:
		
		steering_center = float(row[3])
		# create adjusted steering measurements for the side camera images
		correction = 0.2 # this is a parameter to tune
		steering_left = steering_center + correction
		steering_right = steering_center - correction			
		
		img_center = cv2.imread(DIR_DATA + row[0].strip())
		img_left = cv2.imread(DIR_DATA + row[1].strip())
		img_right = cv2.imread(DIR_DATA + row[2].strip())
		
		#!!!!!!!!!!!!!image = process_image(img_center)
		#img_center = process_image(img_center)
		img_left = process_image(img_left)
		img_right = process_image(img_right)		
    
		# add images and angles to data set
		car_images.extend([img_center, img_left, img_right])
		steering_angles.extend([steering_center, steering_left, steering_right])
		
		img_center_flipped = np.fliplr(img_center)
		img_left_flipped = np.fliplr(img_left)
		img_right_flipped = np.fliplr(img_right)		
		
		steering_center_flipped = -steering_center
		#steering_left_flipped = -steering_center - correction
		#steering_right_flipped = -steering_center + correction
		car_images.append(img_center_flipped)#, img_left_flipped, img_right_flipped])
		steering_angles.append(steering_center_flipped)#, steering_left_flipped, steering_right_flipped])	

	x_train = np.array(car_images)
	y_train = np.array(steering_angles)
	return x_train, y_train


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping

def build_model(keep_prob=0.5):
	"""build a model introduced by NVIDEA"""
	model = Sequential()

	model.add(Lambda(lambda x: x / 255.0 -0.5,input_shape=INPUT_SHAPE))
		
	Lambda(lambda x: tf.image.rgb_to_grayscale(x))
	
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(64,3,3, activation="relu"))
	model.add(Convolution2D(64,3,3, activation="relu"))
	model.add(Dropout(keep_prob))
	model.add(Flatten())	
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

import matplotlib.pyplot as plt

def display_graph(history_object) :
	"""plot the training and validation loss for each epoch"""	
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()

if __name__ == '__main__':
	
	samples = get_logs(DRIVING_LOG)

	from sklearn.model_selection import train_test_split
	train_samples, test_samples = train_test_split(samples, test_size=0.2)

	model = build_model(keep_prob=0.5)
	model.compile(loss='mse', optimizer='adam')

	x_train, y_train = get_data_pair(train_samples)
	x_test, y_test = get_data_pair(test_samples)
		
	history_object = model.fit(x_train, y_train, 
		        batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, 
		        verbose=VERBOSE, validation_split=VALIDATION_SPLIT, shuffle=True,callbacks=[EarlyStopping()])	

	model.save('model.h5')
	score = model.evaluate(x_test, y_test, verbose=VERBOSE)
	print("Score:", score)
	#print('Test accuracy:', score[1])
	
	display_graph(history_object) 