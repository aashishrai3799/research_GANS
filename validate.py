
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
#import cv2 
import numpy as np 
import math 
from inception_resnet_v2 import inception_resnet_v2

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

X_train = np.load('/media/dl/DL/aashish/upscaling/tinyface/X_train.npy')
#Y_image = np.load('/media/ml/Data Disk/upscaling/celebA_data/Y_image.npy')
Y_train = np.load('/media/dl/DL/aashish/upscaling/tinyface/Y_train.npy')

#trainx2 = np.load('/content/drive/My Drive/Drive1(Own)/Arrow224/X_train.npy')
#trainy2 = np.load('/content/drive/My Drive/Drive1(Own)/Arrow224/Y_train.npy')

X_train = (X_train/256)*2 - 1
num_classes = 2570

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
  
Y_train = convert_to_one_hot(Y_train, num_classes).T  

def Upsample2xBlock(x, kernel_size, filters, strides=1):
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
    x = tf.depth_to_space(x, 4)
    x = tf.nn.relu(x)
    return x

def srcnn(X):

    gen1 = tf.layers.conv2d(X, 64, 9, 1, padding = 'same')
    P1 = tf.nn.relu(gen1)    
    
    gen2 = tf.layers.conv2d(P1, 32, 1, 1, padding = 'SAME')
    P2 = tf.nn.relu(gen2)    

    res_in_1 = P2
    rc1 = tf.layers.conv2d(res_in_1, 32, 3, 1, padding = 'SAME')
    rc1 = tf.nn.relu(rc1)
    r1 = tf.layers.conv2d(rc1, 32, 1, 1, padding = 'SAME')
    r1 = tf.nn.relu(r1)
    r1 = tf.layers.conv2d(r1, 32, 1, 1, padding = 'SAME')
    r1 = tf.nn.relu(r1)
    res_out_1 = res_in_1 + r1    

    res_in_2 = res_out_1
    rc2 = tf.layers.conv2d(res_in_2, 32, 3, 1, padding = 'SAME')
    rc2 = tf.nn.relu(rc2)
    r2 = tf.layers.conv2d(rc2, 32, 1, 1, padding = 'SAME')
    r2 = tf.nn.relu(r2)
    r2 = tf.layers.conv2d(r2, 32, 1, 1, padding = 'SAME')
    r2 = tf.nn.relu(r2)
    res_out_2 = res_in_2 + r2 

    res_in_3 = res_out_2
    rc3 = tf.layers.conv2d(res_in_3, 32, 3, 1, padding = 'SAME')
    rc3 = tf.nn.relu(rc3)
    r3 = tf.layers.conv2d(rc3, 32, 1, 1, padding = 'SAME')
    r3 = tf.nn.relu(r3)
    r3 = tf.layers.conv2d(r3, 32, 1, 1, padding = 'SAME')
    r3 = tf.nn.relu(r3)
    res_out_3 = res_in_3 + r3

    res_in_4 = res_out_3
    rc4 = tf.layers.conv2d(res_in_4, 32, 3, 1, padding = 'SAME')
    rc4 = tf.nn.relu(rc4)
    r4 = tf.layers.conv2d(rc4, 32, 1, 1, padding = 'SAME')
    r4 = tf.nn.relu(r4)
    r4 = tf.layers.conv2d(r4, 32, 1, 1, padding = 'SAME')
    r4 = tf.nn.relu(r4)
    res_out_4 = res_in_4 + r4    

    res_in_5 = res_out_4
    rc5 = tf.layers.conv2d(res_in_5, 32, 3, 1, padding = 'SAME')
    rc5 = tf.nn.relu(rc5)
    r5 = tf.layers.conv2d(rc5, 32, 1, 1, padding = 'SAME')
    r5 = tf.nn.relu(r5)
    r5 = tf.layers.conv2d(r5, 32, 1, 1, padding = 'SAME')
    r5 = tf.nn.relu(r5)
    res_out_5 = res_in_5 + r5    

    res_in_1 = res_out_5
    rc1 = tf.layers.conv2d(res_in_1, 32, 3, 1, padding = 'SAME')
    rc1 = tf.nn.relu(rc1)
    r1 = tf.layers.conv2d(rc1, 32, 1, 1, padding = 'SAME')
    r1 = tf.nn.relu(r1)
    r1 = tf.layers.conv2d(r1, 32, 1, 1, padding = 'SAME')
    r1 = tf.nn.relu(r1)
    res_out_1 = res_in_1 + r1    

    res_in_2 = res_out_1
    rc2 = tf.layers.conv2d(res_in_2, 32, 3, 1, padding = 'SAME')
    rc2 = tf.nn.relu(rc2)
    r2 = tf.layers.conv2d(rc2, 32, 1, 1, padding = 'SAME')
    r2 = tf.nn.relu(r2)
    r2 = tf.layers.conv2d(r2, 32, 1, 1, padding = 'SAME')
    r2 = tf.nn.relu(r2)
    res_out_2 = res_in_2 + r2 

    res_in_3 = res_out_2
    rc3 = tf.layers.conv2d(res_in_3, 32, 3, 1, padding = 'SAME')
    rc3 = tf.nn.relu(rc3)
    r3 = tf.layers.conv2d(rc3, 32, 1, 1, padding = 'SAME')
    r3 = tf.nn.relu(r3)
    r3 = tf.layers.conv2d(r3, 32, 1, 1, padding = 'SAME')
    r3 = tf.nn.relu(r3)
    res_out_3 = res_in_3 + r3

    res_in_4 = res_out_3
    rc4 = tf.layers.conv2d(res_in_4, 32, 3, 1, padding = 'SAME')
    rc4 = tf.nn.relu(rc4)
    r4 = tf.layers.conv2d(rc4, 32, 1, 1, padding = 'SAME')
    r4 = tf.nn.relu(r4)
    r4 = tf.layers.conv2d(r4, 32, 1, 1, padding = 'SAME')
    r4 = tf.nn.relu(r4)
    res_out_4 = res_in_4 + r4    

    res_in_5 = res_out_4
    rc5 = tf.layers.conv2d(res_in_5, 32, 3, 1, padding = 'SAME')
    rc5 = tf.nn.relu(rc5)
    r5 = tf.layers.conv2d(rc5, 32, 1, 1, padding = 'SAME')
    r5 = tf.nn.relu(r5)
    r5 = tf.layers.conv2d(r5, 32, 1, 1, padding = 'SAME')
    r5 = tf.nn.relu(r5)
    res_out_5 = res_in_5 + r5    

    regen1 = tf.layers.conv2d(P2, 64, 5, 1, padding = 'same')

    #print('res_out shape:', res_out_2.shape)
    upsam = Upsample2xBlock(P2, kernel_size=3, filters=64)#tf.layers.conv2d_transpose(res_out_3,64,(3, 3),strides=(4,4), padding='same')

    res_in_4 = upsam
    rc4 = tf.layers.conv2d(res_in_4, 32, 3, 1, padding = 'SAME')
    rc4 = tf.nn.relu(rc4)
    r4 = tf.layers.conv2d(rc4, 32, 1, 1, padding = 'SAME')
    r4 = tf.nn.relu(r4)
    r4 = tf.layers.conv2d(r4, 4, 1, 1, padding = 'SAME')
    r4 = tf.nn.relu(r4)
    res_out_4 = res_in_4 + r4    

    res_in_5 = res_out_4
    rc5 = tf.layers.conv2d(res_in_5, 32, 3, 1, padding = 'SAME')
    rc5 = tf.nn.relu(rc5)
    r5 = tf.layers.conv2d(rc5, 32, 1, 1, padding = 'SAME')
    r5 = tf.nn.relu(r5)
    r5 = tf.layers.conv2d(r5, 4, 1, 1, padding = 'SAME')
    r5 = tf.nn.relu(r5)
    res_out_5 = res_in_5 + r5    
    upsam = res_out_5
    #[batch, height, width, in_channels]
    #[height, width, output_channels, in_channels]
    print('upsam shape:', upsam.shape)

    regen1 = tf.layers.conv2d(upsam, 3, 3, 1, padding = 'same')
    A3 = tf.nn.relu(regen1)     
    regen2 = tf.layers.conv2d(A3, 3, 3, 1, padding = 'same')
    
    print('output shape: ', regen2.shape)
    
    return regen2

def inception(X):

    x1 = tf.layers.conv2d(X, 32, 5, 1, padding = 'same')
    A1 = tf.nn.relu(x1)    
    
    gen11 = tf.nn.relu(tf.layers.conv2d(A1, 32, 1, 1, padding = 'SAME'))

    gen21 = tf.nn.relu(tf.layers.conv2d(A1, 32, 1, 1, padding = 'SAME'))
    gen22 = tf.nn.relu(tf.layers.conv2d(gen21, 16, 3, 1, padding = 'SAME'))

    gen31 = tf.nn.relu(tf.layers.conv2d(A1, 16, 1, 1, padding = 'SAME'))
    gen32 = tf.nn.relu(tf.layers.conv2d(gen31, 32, 3, 1, padding = 'SAME'))
    gen33 = tf.nn.relu(tf.layers.conv2d(gen32, 32, 3, 1, padding = 'SAME'))

    l4 = tf.keras.layers.concatenate([gen11, gen22, gen33])

    l5 = tf.nn.relu(l4)    
    l5 = tf.layers.conv2d(l5, 16, 4, 1, padding = 'valid')
    l5 = tf.nn.relu(tf.layers.conv2d(l5, 32, 3, 1, padding = 'SAME'))

    l5 = tf.nn.relu(l5) 
    l5 = tf.layers.conv2d(l5, 16, 4, 2, padding = 'valid')
    l5 = tf.layers.conv2d(l5, 16, 4, 1, padding = 'valid')
    l5 = tf.layers.conv2d(l5, 16, 4, 1, padding = 'valid')
    l5 = tf.layers.conv2d(l5, 16, 4, 1, padding = 'valid')

    A6 = tf.nn.relu(l5)

    A6, _ = inception_resnet_v2(X)
    P_fl = tf.contrib.layers.flatten(A6)
    fc = tf.contrib.layers.fully_connected(P_fl, num_classes, activation_fn = None)
    
    return fc

tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("/media/dl/DL/aashish/upscaling/model/model.ckpt.meta")
                  

costs = []                                                            
t1 = 0
t2 = 0

(m, n_H0, n_W0, n_C0) = X_train.shape  
#(m1, n_H1, n_W1, n_C1) = Y_image.shape             
n_y = Y_train.shape[1]                            

X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = 'X')
#Y_img = tf.placeholder(tf.float32, [None, n_H1, n_W1, n_C1], name = 'Y_img')
Y = tf.placeholder(tf.float32, [None, n_y], name = 'Y')

Z3 = srcnn(X)
Z4 = inception(Z3)
#Z4 = squeeze_net(Z3, num_classes)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    imported_meta.restore(sess, tf.train.latest_checkpoint('/media/dl/DL/aashish/upscaling/model'))
    print("restored")
    '''X_test = np.reshape(X_train[500,:,:,:]/256, (1, 32, 32, 3))
    plt.imshow(X_train[500,:,:,:]/256)
    plt.show()
    Z31 = sess.run([Z3], feed_dict = {X: X_test})
    Z31 = np.asarray(Z31, order= (128, 128, 3))
    Z31 = np.reshape(Z31*25, (128,128,3))
    print(Z31)
    print(np.max(Z31), np.min(Z31))
    #Z31 = Z31 + abs(np.min(Z31))
    #Z31 = Z31/np.max(Z31)
    #print(np.max(Z31), np.min(Z31))

    plt.imshow(Z31)
    plt.show()'''
    #print('Trainable Params:', np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
    correct_prediction = tf.equal(tf.argmax(Z4, 1), tf.argmax(Y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_accuracy = sess.run([correct_prediction], feed_dict = {X: X_train[0:400], Y: Y_train[0:400]})
    #train_accuracy = accuracy.eval({X: X_train[0:50], Y: Y_train[0:50]})
    #, X_t: X_tiny, Y_t:Y_tiny
    tr = train_accuracy.count('False')
    print(tr)
    print("Train Accuracy:", train_accuracy)