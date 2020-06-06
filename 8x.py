
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

print("LOADING DATASET.....")
X_train = np.load('/media/dl/DL/aashish/upscaling/sr_dataset/bgX_train_24.npy')
Y_image = np.load('/media/dl/DL/aashish/upscaling/sr_dataset/bgY_image_192.npy')
Y_train = np.load('/media/dl/DL/aashish/upscaling/sr_dataset/bgY_train_24.npy')
num_classes = 1

X_train = (X_train[:,:,:,:]/255).astype('float32')
Y_image = (Y_image[:,:,:,:]/255).astype('float32')

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
  
Y_train = convert_to_one_hot(Y_train[:], num_classes).T  
print("DATASET LOADED!!")

print(X_train.shape, Y_image.shape, Y_train.shape)

def resblock(X, n):
  res_in_1 = X
  rc1 = tf.layers.conv2d(res_in_1, 32, n, 1, padding = 'SAME')
  rc1 = tf.nn.relu(rc1)
  r1 = tf.layers.conv2d(rc1, 32, n, 1, padding = 'SAME')
  r1 = tf.nn.relu(r1)
  r1 = tf.layers.conv2d(r1, 3, n, 1, padding = 'SAME')
  r1 = tf.nn.relu(r1)
  res_out_1 = res_in_1 + r1  
  return res_out_1

def convrow(X, n):
  for i in range (0, 3):
    X = resblock(X, n)
  return X

def cnn_model(X):

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


minibatch_size = 64
learning_rate = 0.0005
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

Z3 = cnn_model(X)

cost1 = tf.losses.mean_squared_error(Z3, Y_img)
cost1 = tf.reduce_mean(cost1)
#cost2 = tf.nn.softmax_cross_entropy_with_logits(logits = Z4, labels = Y)
#cost2 = tf.reduce_mean(cost2)
cost = cost1 #+ cost2

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)    
init = tf.global_variables_initializer()    
saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    print('Trainable Params:', np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

    for epoch in range(num_epochs):
        
        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size) 
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_image, Y_train, minibatch_size, seed)
        for minibatch in minibatches:

            (minibatch_X, minibatch_Yi, minibatch_Y) = minibatch
            _ , temp_cost, temp_cost1 = sess.run([optimizer, cost, cost1], feed_dict = {X: minibatch_X, Y_img: minibatch_Yi})

            temp_cost1 += temp_cost1 / num_minibatches

        if epoch % 100 == 0:
            psnr = 10*np.log10(np.max(minibatch_Yi)*np.max(minibatch_Yi)/temp_cost1)
            costs.append(temp_cost)
            t2 = time.time()
            print ("Epoch:", epoch, 'Time:', round(t2-t1, 1), 'Total loss:', round(temp_cost, 6), 'SR loss:', round(temp_cost1, 6), 'PSNR:', round(psnr, 6))
            t1 = time.time()

        if epoch % 100 == 0:
            #saver.save(sess, '/media/dl/DL/aashish/upscaling/model/sr_24x24/model.ckpt')
            print('Model Saved...')
            #Z31 = Z3[0].eval(feed_dict = {X: minibatch_X, Y_img: minibatch_Yi})
            #Z31 = Z31**2
            #plt.imsave('/media/dl/DL/aashish/upscaling/sr output/33results/' + str(epoch) + '.jpg', Z31/np.max(Z31), vmin=0, vmax=255)


        if epoch % 500 == 0:
            for aa in range(0, minibatch_X.shape[0]):
                Z31 = Z3[aa].eval(feed_dict = {X: minibatch_X, Y_img: minibatch_Yi})
                Z31 = Z31**2
                plt.imsave('/media/dl/DL/aashish/upscaling/sr output/33results/' + str(epoch) + '_' + str(aa) + '.jpg', Z31/np.max(Z31), vmin=0, vmax=255)


    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    predict_op = tf.argmax(Z4, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('accuracy', accuracy)
    train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
    #test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
    print("Train Accuracy:", train_accuracy)
    #print("Test Accuracy:", test_accuracy)
                 