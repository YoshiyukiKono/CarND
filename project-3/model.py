import csv
import cv2
import numpy as np
import sklearn

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
print(keras.__version__)

DATADIR = "./data/"
DIR_IMG = './data/IMG/'
DIR_DATA = "./data/"

#define the convnet 
class LeNet:
	@staticmethod
	def build(input_shape, classes):
		print(input_shape)
		model = Sequential()
		# CONV => RELU => POOL
	
		model.add(Conv2D(20, #kernel_size=5, #padding="same",
			input_shape=input_shape, nb_row=160, nb_col=320))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# CONV => RELU => POOL
		model.add(Conv2D(50, kernel_size=5, padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# Flatten => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))
 
		# a softmax classifier
		if classes != 1:
			model.add(Dense(classes))
			model.add(Activation("softmax"))
		if classes == 1:
			model.add(Flatten(input_shape=input_shape))
			model.add(Dense(1))
		return model

# network and training
NB_EPOCH = 3#8#6#4# 2#10# 4 #7#20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT=0.2

IMG_ROWS, IMG_COLS = 160, 320 #28, 28 # input image dimensions
NB_CLASSES = 1#10  # number of outputs = number of digits
INPUT_SHAPE = (160, 320, 3)#(1, IMG_ROWS, IMG_COLS)	

IMG_DIM = 3

def getLog():
	lines = []
	with open('./data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		fieldnames = next(reader)
		for line in reader:			
			lines.append(line)
	return lines

def load(lines):  
	images = []
	measurements = []
	for line in lines:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = DIR_IMG + filename
		image = cv2.imread(current_path)
		images.append(image)	
		measurement = float(line[3])
		measurements.append(measurement)	
	x_train = np.array(images)
	y_train = np.array(measurements)
	return x_train, y_train

def equalize_histogram(image):
	copied_image = image.copy()
	copied_image[:,:,0] = cv2.equalizeHist(image[:,:,0])
	copied_image[:,:,1] = cv2.equalizeHist(image[:,:,1])
	copied_image[:,:,2] = cv2.equalizeHist(image[:,:,2])
	return copied_image

def process_image(image):
	if IMG_DIM == 1:
		print("RGB",np.array(image).shape)
		processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		print("GRAY",np.array(processed_image).shape)
	else:
		processed_image = equalize_histogram(image)
	#image_copy = gray_image.copy()	
	#image_copy[:,:,0] = cv2.equalizeHist(gray_image_copy[:,:,0])	
	#return image_copy
	return processed_image#gray_image

def loadWithFlipImage(lines):  
	images = []
	measurements = []
	for line in lines:
		#source_path = line[0]
		#filename = source_path.split('/')[-1]
		#current_path = DIR_IMG + filename
		#image = cv2.imread(current_path)
		image_org = cv2.imread(DIR_DATA + line[0].strip())
		img_left_org = cv2.imread(DIR_DATA + line[1].strip())
		img_right_org = cv2.imread(DIR_DATA + line[2].strip())
		
		image = process_image(image_org)
		img_left = process_image(img_left_org)
		img_right = process_image(img_right_org)		

		images.extend([image, img_left, img_right])
		#images.append(image)
		#images.append(img_left)
		#images.append(img_right)
		
		measurement = float(line[3])
		correction = 0.2 # this is a parameter to tune
		steering_left = measurement + correction
		steering_right = measurement - correction
		
		#measurements.append(measurement)
		#measurements.append(steering_left)
		#measurements.append(steering_right)
		measurements.extend([measurement, steering_left, steering_right])
		
		image_flipped = np.fliplr(image)
		measurement_flipped = -measurement
		images.append(image_flipped)	
		measurements.append(measurement_flipped)		
	
	x_train = np.array(images)
	y_train = np.array(measurements)
	return x_train, y_train



def loadWithMultipleCameras(rows):  
	car_images = []
	steering_angles = []
	for row in rows:
		
		steering_center = float(row[3])
		# create adjusted steering measurements for the side camera images
		correction = 0.2 # this is a parameter to tune
		steering_left = steering_center + correction
		steering_right = steering_center - correction			
		
		#img_center = process_image(np.asarray(cv2.imread(DIR_DATA + row[0].strip())))
		#img_left = process_image(np.asarray(cv2.imread(DIR_DATA + row[1].strip())))
		#img_right = process_image(np.asarray(cv2.imread(DIR_DATA + row[2].strip())))
		
		img_center = cv2.imread(DIR_DATA + row[0].strip())
		img_left = cv2.imread(DIR_DATA + row[1].strip())
		img_right = cv2.imread(DIR_DATA + row[2].strip())
		
		image = process_image(img_center)
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

#from keras.layers.pooling...
def build():
	model = Sequential()
	model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160, 320, IMG_DIM)))
	model.add(Lambda(lambda x: x / 255.0 -0.5))
	#model.add(Lambda(lambda x: x / 255.0 -0.5, input_shape=(160, 320, 3)))
	#model.add(Flatten())
	#model.add(Flatten(input_shape=(160, 320, 3)))
	
	# CONV => RELU => POOL
	model.add(Conv2D(32,3,3))
	model.add(Activation("relu"))
	model.add(Conv2D(32,3,3))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# CONV => RELU => POOL
	model.add(Conv2D(64, 3,3))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Flatten => RELU layers
	model.add(Flatten())	

	model.add(Dense(1))
	return model

def buildNVIDEA(keep_prob=0.5):
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 -0.5,input_shape=(160, 320, IMG_DIM)))
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

def buildForGenerator():
	
	ch, row, col = 3, 80, 320  # Trimmed image format
	
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
		input_shape=(ch, row, col),
		output_shape=(ch, row, col)))	
	#model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160, 320, 3)))
	#model.add(Lambda(lambda x: x / 255.0 -0.5))
	#model.add(Lambda(lambda x: x / 255.0 -0.5, input_shape=(160, 320, 3)))
	#model.add(Flatten())
	model.add(Flatten(input_shape=(160, 320, 3)))
	model.add(Dense(1))
	return model	
	model = Sequential()
	##model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160, 320, 3)))
	#model.add(Lambda(lambda x: x / 255.0 -0.5))
	model.add(Lambda(lambda x: x / 255.0 -0.5, input_shape=(160, 320, 3)))
	#model.add(Flatten())
	model.add(Flatten(input_shape=(160, 320, 3)))
	model.add(Dense(1))
	
	return model

"""
model.add(MaxPooling2D())
model.add(Flattern())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
"""

### plot the training and validation loss for each epoch
import matplotlib.pyplot as plt

def showHistory(history_object) :
	### print the keys contained in the history object
	print(history_object.history.keys())
	
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		#shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			car_images = []
			steering_angles = []
			for row in batch_samples:
				steering_center = float(row[3])
				# create adjusted steering measurements for the side camera images
				correction = 0.2 # this is a parameter to tune
				steering_left = steering_center + correction
				steering_right = steering_center - correction
				
				img_center = cv2.imread(DIR_DATA + row[0].strip())
				img_left = cv2.imread(DIR_DATA + row[1].strip())
				img_right = cv2.imread(DIR_DATA + row[2].strip())
				
				img_center = process_image(img_center)
				img_left = process_image(img_left)
				img_right = process_image(img_right)				
			    
				# add images and angles to data set
				car_images.extend([img_center, img_left, img_right])
				steering_angles.extend([steering_center, steering_left, steering_right])
			    
				img_center_flipped = np.fliplr(img_center)
				img_left_flipped = np.fliplr(img_left)
				img_right_flipped = np.fliplr(img_right)
				steering_center_flipped = -steering_center
				steering_left_flipped = -steering_center - correction
				steering_right_flipped = -steering_center + correction
				car_images.extend([img_center_flipped])#, img_left_flipped, img_right_flipped])
				steering_angles.extend([steering_center_flipped])#, steering_left_flipped, steering_right_flipped])
			x_train = np.array(car_images)
			y_train = np.array(steering_angles)
			yield sklearn.utils.shuffle(x_train, y_train)
		
def generatorPre(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):		
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
	    
def visualize_model(model, file_name="model.png"):
	from keras.utils.visualize_util import plot
	plot(model, to_file=file_name, show_shapes=True)	
	    
##################################
# Main Logic
##################################
samples = getLog()

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)



#x_train, y_train = load(train_samples)
#x_test, y_test = load(validation_samples)

#x_train, y_train = loadWithFlipImage(train_samples)
#x_test, y_test = loadWithFlipImage(validation_samples)

"""
x_train, y_train = loadWithMultipleCameras(train_samples)
x_test, y_test = loadWithMultipleCameras(validation_samples)

print("x_train.shape",x_train.shape)
print("y_train.shape",y_train.shape)
"""

#model = build()
model = buildNVIDEA(keep_prob=0.5)
#model = buildForGenerator()
#model.compile(loss='mse', optimizer='adam')
#model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=NB_EPOCH)

# initialize the optimizer and model
#model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss='mse', optimizer='adam')#"categorical_crossentropy", optimizer=OPTIMIZER,
	#metrics=["accuracy"])

USE_GENERATOR = False
if USE_GENERATOR:
	# compile and train the model using the generator function
	train_generator = generator(train_samples, batch_size=BATCH_SIZE)#32)
	validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)#32)
	history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
		                             validation_data=validation_generator, nb_val_samples=len(validation_samples),
		                             nb_epoch=NB_EPOCH)
else:
	x_train, y_train = loadWithMultipleCameras(train_samples)
	x_test, y_test = loadWithMultipleCameras(validation_samples)
	
	print("x_train.shape",x_train.shape)
	print("y_train.shape",y_train.shape)	
	history_object = model.fit(x_train, y_train, 
		        batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, #epochs=NB_EPOCH, 
		        verbose=VERBOSE, validation_split=VALIDATION_SPLIT, shuffle=True)	
"""
ValueError: output of generator should be a tuple (x, y, sample_weight) or (x, y). Found: None
"""

#score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#score = model.evaluate(x_train, y_train, verbose=VERBOSE)
#print("\nscore:", score)
#print("\nTest score:", score[0])
#print('Test accuracy:', score[1])

showHistory(history_object) 

model.save('model.h5')

visualize_model(model)