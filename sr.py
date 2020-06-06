import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import numpy as np
import random
from PIL import Image
import PIL
from tensorflow.python.framework import ops
import math
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
slim = tf.contrib.slim
import tensorflow.contrib.slim as slim

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print(tf.__version__)

X_train = np.load('/media/dl/DL/aashish/upscaling/sr_dataset/X_train_16.npy')
Y_image = np.load('/media/dl/DL/aashish/upscaling/sr_dataset/Y_image_128.npy')
Y_train = np.load('/media/dl/DL/aashish/upscaling/sr_dataset/Y_train_16.npy')
num_classes = 1
#plt.imshow(X_train[1,:,:,:])
#plt.show()
X_train = (X_train[:,:,:,:]/255).astype('float32')
Y_image = (Y_image[:,:,:,:]/255).astype('float32')

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
  
Y_train = convert_to_one_hot(Y_train[:], num_classes).T  
print(X_train.shape, Y_image.shape, Y_train.shape)

def resblock(X):
  res_in = X
  r3 = tf.layers.conv2d(res_in, 64, 3, 1, padding = 'SAME')
  r3 = tf.nn.relu(r3)
  r3 = tf.layers.conv2d(r3, 64, 1, 1, padding = 'SAME')
  r3 = tf.nn.relu(r3)
  r3 = tf.layers.conv2d(r3, 64, 1, 1, padding = 'SAME')
  r3 = tf.nn.relu(r3)

  r5 = tf.layers.conv2d(res_in, 64, 5, 1, padding = 'SAME')
  r5 = tf.nn.relu(r5)
  r5 = tf.layers.conv2d(r5, 64, 1, 1, padding = 'SAME')
  r5 = tf.nn.relu(r5)
  r5 = tf.layers.conv2d(r5, 64, 1, 1, padding = 'SAME')
  r5 = tf.nn.relu(r5)

  r7 = tf.layers.conv2d(res_in, 64, 1, 1, padding = 'SAME')
  r7 = tf.nn.relu(r7)
  r7 = tf.layers.conv2d(r7, 64, 1, 1, padding = 'SAME')
  r7 = tf.nn.relu(r7)
  r7 = tf.layers.conv2d(r7, 64, 1, 1, padding = 'SAME')
  r7 = tf.nn.relu(r7)

  concatenate = tf.concat([r3, r5, r7], 3)
  out = tf.layers.conv2d(concatenate, 64, 1, 1, padding = 'SAME')
  out = out + X

  return out


def my_sr(X):
  X = tf.layers.conv2d(X, 64, 7, 1, padding = 'SAME')
  rb1 = resblock(X)
  rb2 = resblock(rb1)
  rb3 = resblock(rb2)
  rb4 = resblock(rb3)
  rb5 = resblock(rb4)
  rb6 = resblock(rb5)
  rb6 = rb6 + X
  rb7 = Upsample2xBlock(rb6)
  #rb6 = tf.layers.conv2d(rb6, 64, 1, 1, padding = 'SAME')
  rb8 = resblock(rb7)
  rb9 = resblock(rb8)
  rb9 = rb9 + rb7
  rb10 = Upsample2xBlock(rb9)
  #rb8 = tf.layers.conv2d(rb8, 64, 1, 1, padding = 'SAME')
  rb11 = resblock(rb10)
  rb12 = resblock(rb11)
  rb12 = rb12 + rb10

  #concatenate = tf.concat([rb1, rb2, rb3, rb4, rb5, rb6, rb7, rb8, rb9, rb10], 3)

  #Y = tf.layers.conv2d(rb10, 64, 1, 1, padding = 'SAME')
  upsam = Upsample2xBlock(rb12)
  print('upsam shape:', upsam.shape)

  Y = tf.layers.conv2d(upsam, 64, 3, 1, padding = 'SAME')
  Y = tf.layers.conv2d(Y, 3, 7, 1, padding = 'SAME')

  return Y

def Upsample2xBlock(x):
    size = tf.shape(x)
    h = size[1]
    w = size[2]
    x = tf.image.resize_nearest_neighbor(x, size=[h * 2, w * 2],  align_corners=False,name=None)
    x = tf.layers.conv2d(x, 64, 3, 1, padding='same')
    x = tf.nn.relu(x)
    return x

def random_mini_batches(X, Y_img, Y, mini_batch_size = 64, seed = 10):
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y_img = Y_img[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y_img = shuffled_Y_img[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y_img, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y_img = shuffled_Y_img[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y_img, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

minibatch_size = 128
learning_rate = 0.005
num_epochs = 100001
seed = 10
print('X_train shape', X_train.shape)  
print('Y_train shape', Y_train.shape)  
print('Learning rate:', learning_rate)

(m, n_H0, n_W0, n_C0) = X_train.shape  
(m1, n_H1, n_W1, n_C1) = Y_image.shape             
n_y = Y_train.shape[1]                            
costs = []                                                            
t1 = 0
t2 = 0

X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = 'X')
Y_img = tf.placeholder(tf.float32, [None, n_H1, n_W1, n_C1], name = 'Y_img')
Y = tf.placeholder(tf.float32, [None, n_y], name = 'Y')

Z3 = my_sr(X)

cost1 = tf.losses.mean_squared_error(Z3, Y_img)
cost1 = tf.reduce_mean(cost1)

cost = cost1

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)    
init = tf.global_variables_initializer()    
saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    print('Trainable Params:', np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

    for epoch in range(num_epochs):
        
        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_image, Y_train, minibatch_size, seed)
        for minibatch in minibatches:

            (minibatch_X, minibatch_Yi, minibatch_Y) = minibatch
            _ , temp_cost, temp_cost1 = sess.run([optimizer, cost, cost1], feed_dict = {X: minibatch_X, Y_img: minibatch_Yi})

            #temp_cost += temp_cost / num_minibatches
            temp_cost1 += temp_cost1 / num_minibatches

        if epoch % 1 == 0:
            psnr = 10*np.log10(np.max(minibatch_Yi)*np.max(minibatch_Yi)/temp_cost1)
            costs.append(temp_cost)
            t2 = time.time()
            print ("Epoch:", epoch, 'Time:', round(t2-t1, 1), 'Total loss:', round(temp_cost, 6), 'SR loss:', round(temp_cost1, 6), 'PSNR:', round(psnr, 6))
            t1 = time.time()

        if epoch % 5 == 0:
            #print('Saving Model...')
            saver.save(sess, '/media/dl/DL/aashish/upscaling/model/sr/model.ckpt')
            print('Model Saved...')
            Z31 = Z3[0].eval(feed_dict = {X: minibatch_X, Y_img: minibatch_Yi})
            #plt.imshow(Z31)
            #plt.show()
            #print(np.max(Z31), np.min(Z31))
            Z31 = Z31**2
            plt.imsave('/media/dl/DL/aashish/upscaling/sr output/' + str(epoch) + '.jpg', Z31/np.max(Z31), vmin=0, vmax=255)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

