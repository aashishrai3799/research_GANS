
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
#import cv2 
import numpy as np 
import math 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

X_train = np.load('/media/ml/Data Disk/upscaling/celebA_data/X_train.npy')
Y_image = np.load('/media/ml/Data Disk/upscaling/celebA_data/Y_image.npy')
Y_train = np.load('/media/ml/Data Disk/upscaling/celebA_data/Y_train.npy')

#trainx2 = np.load('/content/drive/My Drive/Drive1(Own)/Arrow224/X_train.npy')
#trainy2 = np.load('/content/drive/My Drive/Drive1(Own)/Arrow224/Y_train.npy')

X_train = (X_train/255).astype('float16')
Y_image = (Y_image/255).astype('float16')

#X_celeb = X_celeb.astype('float16')
#X_celeb = np.resize(X_celeb, (train_size, 160, 160, 3))

#X_test = testx/255
#X_test = X_test.astype('float16')
#X_test = np.resize(X_test, (test_size, 64, 64, 3))

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
  
Y_train = convert_to_one_hot(Y_train, 1894).T  
#Y_test = convert_to_one_hot(testy, 8).T

print(X_train.shape, Y_image.shape, Y_train.shape)
#print(X_test.shape, Y_test.shape)


def srcnn(X):

    gen1 = tf.layers.conv2d(X, 64, 5, 1, padding = 'same')
    P1 = tf.nn.relu(gen1)    
    
    gen2 = tf.layers.conv2d(P1, 64, 3, 1, padding = 'SAME')
    P2 = tf.nn.relu(gen2)    

    res_in_1 = A2
    rc1 = tf.layers.conv2d(res_in, 32, 3, 1, padding = 'SAME')
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

    print('res_out shape:', res_out.shape)
    upsam = tf.layers.conv2d_transpose(res_out,64,(3, 3),strides=(4,4), padding='same')
    #[batch, height, width, in_channels]
    #[height, width, output_channels, in_channels]
    print('upsam shape:', upsam.shape)

    regen1 = tf.layers.conv2d(upsam, 64, 3, 1, padding = 'same')
    A3 = tf.nn.relu(regen1)     
    regen2 = tf.layers.conv2d(A3, 3, 3, 1, padding = 'same')
    
    print('output shape: ', regen2.shape)
    
    return regen2

def inception(X, num_op):

    A1 = tf.nn.relu(X)    
    
    gen11 = tf.layers.conv2d(A1, 64, 1, 1, padding = 'SAME')

    gen21 = tf.layers.conv2d(A1, 32, 1, 1, padding = 'SAME')
    gen22 = tf.layers.conv2d(gen21, 16, 3, 1, padding = 'SAME')

    gen31 = tf.layers.conv2d(A1, 16, 1, 1, padding = 'SAME')
    gen32 = tf.layers.conv2d(gen31, 32, 3, 1, padding = 'SAME')
    gen33 = tf.layers.conv2d(gen32, 64, 3, 1, padding = 'SAME')

    l4 = tf.keras.layers.concatenate([gen11, gen22, gen33])
    l5 = tf.keras.layers.concatenate([l4, A1])
    
    A6 = tf.nn.relu(l5)    
    
    P_fl = tf.contrib.layers.flatten(A6)
    fc = tf.contrib.layers.fully_connected(P_fl, num_op, activation_fn = None)
    
    return fc

def random_mini_batches(X, Y, mini_batch_size = 128, seed = 10):
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


learning_rate = 0.0001
num_epochs = 500
minibatch_size = 128
seed = 10


(m, n_H0, n_W0, n_C0) = X_celeb.shape  
(m1, n_H1, n_W1, n_C1) = X_tiny.shape             
n_y = Y_celeb.shape[1]       
n_y1 = Y_tiny.shape[1]                            

costs = []                                                            
t1 = 0
t2 = 0

(m, n_H0, n_W0, n_C0) = X_train.shape  
(m1, n_H1, n_W1, n_C1) = Y_image.shape             
n_y = Y_train.shape[1]                            
costs = []                                                            
t1 = 0
t2 = 0

X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = 'X')
Y_img = tf.placeholder(tf.float32, [None, n_H1, n_W1, n_C1], name = 'Y_img')
Y = tf.placeholder(tf.float32, [None, n_y], name = 'Y')

Z3 = srcnn(X)
Z4 = inception(Z3)

cost1 = tf.losses.mean_squared_error(Z3, Y_img)
cost1 = tf.reduce_mean(cost1)

cost2 = tf.nn.softmax_cross_entropy_with_logits(logits = Z4, labels = Y)
cost2 = tf.reduce_mean(cost2)

cost = 0.003*cost1 + cost2



optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)    
init = tf.global_variables_initializer()    
saver = tf.train.Saver()

with tf.Session() as sess:
        
    # Run the initialization
    sess.run(init)

    for epoch in range(num_epochs):

        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_celeb, Y_celeb, minibatch_size, seed)
        minibatches2 = random_mini_batches(X_tiny, Y_tiny, minibatch_size, seed)

        ii = 0
        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            (minibatch_Xt, minibatch_Yt) = minibatches2[ii]
            if ii == len(minibatches2)-1:
                ii = 0
            ii += 1
            _ , temp_cost = sess.run([optimizer, cost], feed_dict = {X_c: minibatch_X, Y_c: minibatch_Y, X_t: minibatch_Xt, Y_t: minibatch_Yt})

        # Print the cost every epoch
        if epoch % 1 == 0:
            t2 = time.time()
            print ("Epoch:", epoch, 'Time:', round(t2-t1, 1), 'Total loss:', round(temp_cost, 6))
            t1 = time.time()
            costs.append(temp_cost)

        if epoch % 20 == 0:  
            saver.save(sess, '/media/ml/Data Disk/upscaling/model/model.ckpt')
            file = open('/media/ml/Data Disk/upscaling/model/cost.txt', 'w')
            file.write(str(cost))
            file.close()
            print('Model Saved...')


    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Calculate the correct predictions
    predict_op = tf.argmax(Z_t, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y_t, 1))
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('accuracy', accuracy)
    train_accuracy = accuracy.eval({X: X_tiny, Y: Y_tiny})
    #test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
    print("Train Accuracy:", train_accuracy)
    #print("Test Accuracy:", test_accuracy)
                 

