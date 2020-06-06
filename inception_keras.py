from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten, Conv2DTranspose

import numpy as np 
from keras.datasets import cifar10 
from keras import backend as K 
from keras.utils import np_utils

import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler

print(tf.__version__)

X_train = np.load('/media/ml/Data Disk/upscaling/tinyface/X_train64.npy')
Y_train = np.load('/media/ml/Data Disk/upscaling/tinyface/Y_train64.npy')

train_size = 2117
#test_size = 70

train_images = np.resize(X_train, (train_size, 160, 160, 3))
train_labels = np.resize(Y_train, (train_size))

#test_images = np.resize(X_train, (test_size, 160, 160, 3)) / 255
#test_labels = np.resize(Y_test, (test_size))

print(train_images.shape, train_labels.shape)
#print(test_images.shape, test_labels.shape)


in_size = 16

x = keras.layers.Input(shape=(160, 160, 3))

x1 = keras.activations.relu(x)    

gen11 = keras.layers.Conv2D(16, (1, 1), padding = 'SAME', activation = 'relu')(x1)

gen21 = keras.layers.Conv2D(16, (1, 1), padding = 'SAME', activation = 'relu')(x1)
gen22 = keras.layers.Conv2D(16, (3, 3), padding = 'SAME', activation = 'relu')(gen21)

gen31 = keras.layers.Conv2D(16, (1, 1), padding = 'SAME', activation = 'relu')(x1)
gen32 = keras.layers.Conv2D(16, (3, 3), padding = 'SAME', activation = 'relu')(gen31)
gen33 = keras.layers.Conv2D(16, (3, 3), padding = 'SAME', activation = 'relu')(gen32)

l4 = tf.keras.layers.concatenate([gen11, gen22, gen33])
l5 = tf.keras.layers.concatenate([l4, x1])

A6 = keras.activations.relu(l5)    

x4 = keras.layers.Flatten()(A6)
x5 = keras.layers.Dense(294, activation = 'softmax')(x4)



model = keras.models.Model([x], x5)
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=50, batch_size = 32)

model.save('/media/ml/Data Disk/upscaling/tinyface/tinymodel.h5')