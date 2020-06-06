import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
#import cv2 
import numpy as np 
import math 
from sklearn.preprocessing import normalize
from inception_resnet_v2 import inception_resnet_v2
print(tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
pathin = '/media/dl/DL/aashish/upscaling/sr_dataset/'
X_train = np.load(pathin + 'X_train_24.npy')
Y_image = np.load(pathin + 'Y_image_192.npy')
Y_train = np.load(pathin + 'Y_train_24.npy')
num_classes = 1000
kf = 1.2
X_train = (X_train[:,:,:,:]/255).astype('float32')
Y_image = (Y_image[:,:,:,:]/255).astype('float32')

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
  
Y_train = convert_to_one_hot(Y_train[:], num_classes).T  

print(X_train.shape, Y_image.shape, Y_train.shape)

def srcnn(X):

    gen1 = tf.layers.conv2d(X, 64, 9, 1, padding = 'same')
    P1 = tf.nn.relu(gen1)    
    
    gen2 = tf.layers.conv2d(P1, 64, 1, 1, padding = 'SAME')
    P2 = tf.nn.relu(gen2)    

    res_in_1 = P2
    rc1 = tf.layers.conv2d(res_in_1, 64, 3, 1, padding = 'SAME')
    rc1 = tf.nn.relu(rc1)
    r1 = tf.layers.conv2d(rc1, 64, 1, 1, padding = 'SAME')
    r1 = tf.nn.relu(r1)
    r1 = tf.layers.conv2d(r1, 64, 1, 1, padding = 'SAME')
    r1 = tf.nn.relu(r1)
    res_out_1 = res_in_1 + r1    

    res_in_2 = res_out_1
    rc2 = tf.layers.conv2d(res_in_2, 64, 3, 1, padding = 'SAME')
    rc2 = tf.nn.relu(rc2)
    r2 = tf.layers.conv2d(rc2, 64, 1, 1, padding = 'SAME')
    r2 = tf.nn.relu(r2)
    r2 = tf.layers.conv2d(r2, 64, 1, 1, padding = 'SAME')
    r2 = tf.nn.relu(r2)
    res_out_2 = res_in_2 + r2 

    res_in_3 = res_out_2
    rc3 = tf.layers.conv2d(res_in_3, 64, 3, 1, padding = 'SAME')
    rc3 = tf.nn.relu(rc3)
    r3 = tf.layers.conv2d(rc3, 64, 1, 1, padding = 'SAME')
    r3 = tf.nn.relu(r3)
    r3 = tf.layers.conv2d(r3, 64, 1, 1, padding = 'SAME')
    r3 = tf.nn.relu(r3)
    res_out_3 = res_in_3 + r3

    res_in_4 = res_out_3
    rc4 = tf.layers.conv2d(res_in_4, 64, 3, 1, padding = 'SAME')
    rc4 = tf.nn.relu(rc4)
    r4 = tf.layers.conv2d(rc4, 64, 1, 1, padding = 'SAME')
    r4 = tf.nn.relu(r4)
    r4 = tf.layers.conv2d(r4, 64, 1, 1, padding = 'SAME')
    r4 = tf.nn.relu(r4)
    res_out_4 = res_in_4 + r4    

    res_in_5 = res_out_4
    rc5 = tf.layers.conv2d(res_in_5, 64, 3, 1, padding = 'SAME')
    rc5 = tf.nn.relu(rc5)
    r5 = tf.layers.conv2d(rc5, 64, 1, 1, padding = 'SAME')
    r5 = tf.nn.relu(r5)
    r5 = tf.layers.conv2d(r5, 64, 1, 1, padding = 'SAME')
    r5 = tf.nn.relu(r5)
    res_out_5 = res_in_5 + r5    

    res_in_6 = res_out_5
    rc6 = tf.layers.conv2d(res_in_6, 64, 3, 1, padding = 'SAME')
    rc6 = tf.nn.relu(rc6)
    r6 = tf.layers.conv2d(rc6, 64, 1, 1, padding = 'SAME')
    r6 = tf.nn.relu(r6)
    r6 = tf.layers.conv2d(r6, 64, 1, 1, padding = 'SAME')
    r6 = tf.nn.relu(r6)
    res_out_6 = res_in_6 + r6 

    upsam = Upsample2xBlock(res_out_6, kernel_size=3, filters=64*4)

    res_in_5 = upsam
    rc5 = tf.layers.conv2d(res_in_5, 64, 3, 1, padding = 'SAME')
    rc5 = tf.nn.relu(rc5)
    r5 = tf.layers.conv2d(rc5, 64, 1, 1, padding = 'SAME')
    r5 = tf.nn.relu(r5)
    r5 = tf.layers.conv2d(r5, 64, 1, 1, padding = 'SAME')
    r5 = tf.nn.relu(r5)
    res_out_5 = res_in_5 + r5    

    res_in_6 = res_out_5
    rc6 = tf.layers.conv2d(res_in_6, 64, 3, 1, padding = 'SAME')
    rc6 = tf.nn.relu(rc6)
    r6 = tf.layers.conv2d(rc6, 64, 1, 1, padding = 'SAME')
    r6 = tf.nn.relu(r6)
    r6 = tf.layers.conv2d(r6, 64, 1, 1, padding = 'SAME')
    r6 = tf.nn.relu(r6)
    res_out_6 = res_in_6 + r6 

    res_in_6 = res_out_6
    rc6 = tf.layers.conv2d(res_in_6, 64, 3, 1, padding = 'SAME')
    rc6 = tf.nn.relu(rc6)
    r6 = tf.layers.conv2d(rc6, 64, 1, 1, padding = 'SAME')
    r6 = tf.nn.relu(r6)
    r6 = tf.layers.conv2d(r6, 64, 1, 1, padding = 'SAME')
    r6 = tf.nn.relu(r6)
    res_out_6 = res_in_6 + r6 

    res_in_6 = res_out_6
    rc6 = tf.layers.conv2d(res_in_6, 64, 3, 1, padding = 'SAME')
    rc6 = tf.nn.relu(rc6)
    r6 = tf.layers.conv2d(rc6, 64, 1, 1, padding = 'SAME')
    r6 = tf.nn.relu(r6)
    r6 = tf.layers.conv2d(r6, 64, 1, 1, padding = 'SAME')
    r6 = tf.nn.relu(r6)
    res_out_6 = res_in_6 + r6 

    upsam = Upsample2xBlock(res_out_6, kernel_size=3, filters=64*4)

    res_in_5 = upsam
    rc5 = tf.layers.conv2d(res_in_5, 64, 3, 1, padding = 'SAME')
    rc5 = tf.nn.relu(rc5)
    r5 = tf.layers.conv2d(rc5, 64, 1, 1, padding = 'SAME')
    r5 = tf.nn.relu(r5)
    r5 = tf.layers.conv2d(r5, 64, 1, 1, padding = 'SAME')
    r5 = tf.nn.relu(r5)
    res_out_5 = res_in_5 + r5    

    res_in_6 = res_out_5
    rc6 = tf.layers.conv2d(res_in_6, 64, 3, 1, padding = 'SAME')
    rc6 = tf.nn.relu(rc6)
    r6 = tf.layers.conv2d(rc6, 64, 1, 1, padding = 'SAME')
    r6 = tf.nn.relu(r6)
    r6 = tf.layers.conv2d(r6, 64, 1, 1, padding = 'SAME')
    r6 = tf.nn.relu(r6)
    res_out_6 = res_in_6 + r6 

    upsam = Upsample2xBlock(res_out_6, kernel_size=3, filters=64*4)

    print('upsam shape:', upsam.shape)

    regen1 = tf.layers.conv2d(upsam, 64, 3, 1, padding = 'same')
    A3 = tf.nn.relu(regen1)     
    regen2 = tf.layers.conv2d(A3, 3, 3, 1, padding = 'same')
    
    print('output shape: ', regen2.shape)
    
    return regen2


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

def Upsample2xBlock(x, kernel_size, filters, strides=1):
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
    x = tf.depth_to_space(x, 2)
    x = tf.nn.relu(x)
    return x

tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("/media/dl/DL/aashish/upscaling/model/sr/model.ckpt.meta")
                  

costs = []                                                            
t1 = 0
t2 = 0
seed = 10
minibatch_size = 512
psnr = 0
psnr_t = 0

(m, n_H0, n_W0, n_C0) = X_train.shape  
(m1, n_H1, n_W1, n_C1) = Y_image.shape             
n_y = Y_train.shape[1]                            

X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = 'X')
Y_img = tf.placeholder(tf.float32, [None, n_H1, n_W1, n_C1], name = 'Y_img')
Y = tf.placeholder(tf.float32, [None, n_y], name = 'Y')

Z3 = srcnn(X)

cost1 = tf.losses.mean_squared_error(Z3, Y_img)
cost1 = tf.reduce_mean(cost1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    imported_meta.restore(sess, tf.train.latest_checkpoint('/media/dl/DL/aashish/upscaling/model/sr'))
    print("restored")

    minibatch_cost = 0.
    num_minibatches = int(m / minibatch_size) 
    seed = seed + 1
    minibatches = random_mini_batches(X_train, Y_image, Y_train, minibatch_size, seed)
    for minibatch in minibatches:

        (minibatch_X, minibatch_Yi, minibatch_Y) = minibatch
        pred = sess.run([Z3], feed_dict = {X: minibatch_X, Y_img: minibatch_Yi})
        #pred = pred[4:pred.shape[0]-4, 4:pred.shape[1]-4]
        #pred = minibatch_Yi**kf
        mse = np.mean((pred - minibatch_Yi)**2)
        psnr = abs(10*np.log10(np.max(pred)**2/mse))
        psnr_t += psnr
        print("PSNR:", psnr)

    #plt.imshow(pred[0,0,:,:,:]/255)
    #plt.show()



    print("Overall PSNR:", psnr_t/len(minibatches))