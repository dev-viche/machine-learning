#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:01:20 2017

@author: viche
"""

# Data Preprocessing
# Structure the folders for the pictures - readable by by Keras library.
# Folder structure: parent -> training_set & test_set -> individual categories

# ---------------------------------
# Build the CNN model --- starts ---

# Import the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Initialize the CNN
classifier = Sequential()

# Add Layer - Convolution
# convo: 32=number of filters, 3*3 pixel grid filter
# input_shape: 3=color pic; 256*256 pixels
classifier.add(Convolution2D(32,kernel_size=(3,3),input_shape=(64,64,3),activation='relu'))

# Add Layer - Max Pooling
# pool size: 2*2 pixel grid
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add Layer - Convolution - 2nd
# convo: 32=number of filters, 3*3 pixel grid filter
# input_shape: 3=color pic; 256*256 pixels
classifier.add(Convolution2D(32,kernel_size=(3,3),activation='relu'))

# Add Layer - Max Pooling - 2nd
# pool size: 2*2 pixel grid
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Add Layer - Flattern
classifier.add(Flatten())

# Add Layer - Full Connected
# Hidden Layer
classifier.add(Dense(128, activation='relu'))
# Output Layer
classifier.add(Dense(1, activation='sigmoid'))

# Compile the model
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('cnn_dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory('cnn_dataset/test_set',
                    target_size=(64, 64),
                    batch_size=32,
                    class_mode='binary')

classifier.fit_generator(train_generator,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=validation_generator,
                        validation_steps=2000)


