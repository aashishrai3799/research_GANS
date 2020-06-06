import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import os
import cv2 
import numpy as np 
import math 

print(tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
path = '/media/dl/DL/aashish/upscaling/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/'
#path = '/media/dl/DL/aashish/upscaling/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/'

classes = os.listdir(path)
train_size = 13233
file_size = 192
file_size2 = 24
print(classes)
index = 0
counter = 1
X_train = np.zeros((train_size, int(file_size2), int(file_size2), 3), dtype = 'uint8')
Y_image = np.zeros((train_size, int(file_size), int(file_size), 3), dtype = 'uint8')
#Y_image = original image
#X_train = downscaled image
Y_train = np.zeros((train_size), dtype = 'uint8')
for i in classes:

    print('class: ', i)
    files = os.listdir(str(path) + str(i))

    for k in files:
        img = Image.open(str(path) + str(i) + '/' + str(k))
        img.load
        #print(img.size)
        img = img.resize((int(file_size), int(file_size)), Image.ANTIALIAS)
        img_down = img.resize((int(file_size2), int(file_size2)), Image.ANTIALIAS)
        npimg = np.asarray( img, dtype="uint8" )
        npimg_down = np.asarray( img_down, dtype="uint8" )
        X_train[counter,:,:,:] = npimg_down
        Y_image[counter,:,:,:] = npimg
        Y_train[counter] = classes.index(i)
        '''
        X_train[counter+1,:,:,:] = npimg_down
        Y_image[counter+1,:,:,:] = npimg
        Y_train[counter+1] = classes.index(i)
        
        X_train[counter+2,:,:,:] = npimg_down
        Y_image[counter+2,:,:,:] = npimg
        Y_train[counter+2] = classes.index(i)
        
        X_train[counter+3,:,:,:] = npimg_down
        Y_image[counter+3,:,:,:] = npimg
        Y_train[counter+3] = classes.index(i)
        '''
        counter += 1
        if counter==train_size:
            break
  
print('Total Images', counter)
save_path = '/media/dl/DL/aashish/upscaling/sr_dataset/'
np.save(save_path + 'l_X_train_' + str(file_size2) + '.npy', X_train)
np.save(save_path + 'l_Y_train_' + str(file_size2) + '.npy', Y_train)
np.save(save_path + 'l_Y_image_' + str(file_size) + '.npy', Y_image)
print(X_train.shape, Y_image.shape, Y_train.shape)

