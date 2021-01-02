#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 11:32:00 2021

@author: nabeelhussain
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkContext
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

start_time = datetime.now()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

mainDIR = os.listdir('/scratch/jortberg.e/FinalProject/chest_xray/')
print(mainDIR)

train_folder= '/scratch/jortberg.e/FinalProject/chest_xray/train/'
val_folder = '/scratch/jortberg.e/FinalProject/chest_xray/val/'
test_folder = '/scratch/jortberg.e/FinalProject/chest_xray/test/'

tf.random.set_seed(42)


# train set
os.listdir(train_folder)
train_normal_dir = train_folder+'NORMAL/'
train_pneu_dir = train_folder+'PNEUMONIA/'
val_normal_dir = val_folder+'NORMAL/'
val_pneu_dir = val_folder+'PNEUMONIA/'
test_normal_dir = test_folder+'NORMAL/'
test_pneu_dir = test_folder+'PNEUMONIA/'
print(len(os.listdir(train_normal_dir)))
print(len(os.listdir(train_pneu_dir)))

cnn = Sequential()
cnn.add(Conv2D(filters=16, kernel_size=(7,7), padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=(3,3)))
cnn.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dense(activation = 'sigmoid', units = 1))



# Compile the Neural network
optimizer = Adam(lr = 0.0001, decay=1e-5)
cnn.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
callback = EarlyStopping(monitor='loss', patience=6)


batchsize = 32

# Fitting the CNN to the images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.

training_set = train_datagen.flow_from_directory('/scratch/jortberg.e/FinalProject/chest_xray/train',
                                                 target_size = (196, 196),
                                                 batch_size = batchsize,
                                                 color_mode = 'grayscale',
                                                 class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('/scratch/jortberg.e/FinalProject/chest_xray/val/',
    target_size=(196, 196),
    batch_size = batchsize,
    color_mode = 'grayscale',
    class_mode='binary')

test_set = test_datagen.flow_from_directory('/scratch/jortberg.e/FinalProject/chest_xray/test',
                                            target_size = (196, 196),
                                            batch_size = batchsize,
                                            color_mode = 'grayscale',
                                            class_mode = 'binary')

train_start_time = datetime.now()
cnn_model = cnn.fit(training_set,
                    epochs = 10,
                    validation_data = validation_generator,
                    validation_steps = 624,
                    callbacks = [callback],
                    verbose = 1
                    )
train_end_time = datetime.now()
print('Train Time elapsed', train_end_time - train_start_time)
test_accu = cnn.evaluate_generator(test_set,steps=624)
print('The testing accuracy is :',test_accu[1]*100, '%')

end_time = datetime.now()

print('\nStart time', start_time)
print('End time', end_time)
print('Time elapsed', end_time - start_time)