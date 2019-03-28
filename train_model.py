# Python file to train the model. 
# Takes in one argument, the name the user wants to save
# the trained model as.

import os
import sys
import math
import tensorflow as tf
from PIL import Image
from PIL import ImageFile
from keras.preprocessing.image import ImageDataGenerator
from keras import *
from keras.applications import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import matplotlib.pyplot as plt

def main():
	# Prevent errors
	Image.MAX_IMAGE_PIXELS = None
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	
	# Surpress warnings
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	save_name = sys.argv[1]
	
	base_dir = os.getcwd() 
	train_dir = os.path.join(base_dir, 'train')
	validation_dir = os.path.join(base_dir, 'validate')
	test_dir = os.path.join(base_dir, 'test')
	
	# Create the data_generators for the train and validation data
	train_generator = create_datagen(train_dir)
	validation_generator = create_datagen(validation_dir)
	
	# Create the transfer learning model
	model = create_transfer_learning_model()
	
	# Train model, takes in four inputs, and saves the best model as 'save_name'
	history = train(model, train_generator, validation_generator, save_name)
	
	# Plot a graph of the accuracy and loss
	plot_graph(history)
	
	# Check test accuracy
	test_model(save_name, test_dir)
	

# Function to test model
def test_model(save_name, test_dir):
	model = models.load_model(save_name)
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_directory(
			test_dir,
			target_size=(331, 331),
			batch_size=16,
			class_mode='categorical')

	test_loss, test_acc = model.evaluate_generator(test_generator, steps=math.ceil(1210/16))
	print('test acc:', test_acc)
	

# Function to plot the graph of accuracy and loss over the epoches
def plot_graph(history):

	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(len(acc))

	plt.plot(epochs, acc, 'b', label='Training acc')
	plt.plot(epochs, val_acc, 'r', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()

	plt.figure()

	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()

	plt.show()
		

# Function to train the model		
def train(model,train_generator, validation_generator,save_name):
	
	checkpoint = ModelCheckpoint(save_name,monitor = 'val_loss', save_best_only=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min')
	
	history = model.fit_generator(
      train_generator,
      steps_per_epoch=math.ceil(5642/16),
      epochs=25, 
	  callbacks = [checkpoint,reduce_lr],
      validation_data=validation_generator,
      validation_steps=math.ceil(1210/16))
	
	return history
		
# Function to create the model
def create_transfer_learning_model():
	
	base_model = NASNetLarge(weights='imagenet', include_top=False) # Remove top layer
	
	# Create simple custom layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.5)(x)
	predictions = Dense(2, activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=predictions)

	# Lock the weights of the hidden layers
	for layer in base_model.layers:
		layer.trainable = False

	opt = Adam() # Use Adam as the optimizer
	model.compile(optimizer=opt,
				  loss='binary_crossentropy',
				  metrics=['accuracy'])
	
	return model
	
	
	
# Function to create the data_generator, 
# with input as the directory of the images
def create_datagen(dir):  

	# All images will be rescaled by 1./255
	datagen = ImageDataGenerator(rotation_range=40,
			width_shift_range=0.3,
			height_shift_range=0.3,
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			vertical_flip=True,
			fill_mode='nearest')

	generator = datagen.flow_from_directory(
			dir, # This is the input, target directory
			target_size=(331, 331), # All images will be resized to 331x331
			batch_size=16,
			class_mode='categorical')
			
	return generator
	

if __name__ == '__main__':
	main()