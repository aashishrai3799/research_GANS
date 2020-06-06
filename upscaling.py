
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

X_train = np.load('/media/dl/DL/aashish/upscaling/celebA_data/X_train.npy')
Y_image = np.load('/media/dl/DL/aashish/upscaling/celebA_data/Y_image.npy')
Y_train = np.load('/media/dl/DL/aashish/upscaling/celebA_data/Y_train.npy')

#trainx2 = np.load('/content/drive/My Drive/Drive1(Own)/Arrow224/X_train.npy')
#trainy2 = np.load('/content/drive/My Drive/Drive1(Own)/Arrow224/Y_train.npy')

X_train = ((X_train/255)*2-1).astype('float16')
Y_image = ((Y_image/255)*2-1).astype('float16')
num_classes = 5
#X_celeb = X_celeb.astype('float16')
#X_celeb = np.resize(X_celeb, (train_size, 160, 160, 3))

#X_test = testx/255
#X_test = X_test.astype('float16')
#X_test = np.resize(X_test, (test_size, 64, 64, 3))

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
  
Y_train = convert_to_one_hot(Y_train, num_classes).T  
#Y_test = convert_to_one_hot(testy, 8).T

print(X_train.shape, Y_image.shape, Y_train.shape)
#print(X_test.shape, Y_test.shape)

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

    

    A6, _ = inception_resnet_v2(X)
    P_fl = tf.contrib.layers.flatten(A6)
    fc = tf.contrib.layers.fully_connected(P_fl, num_classes, activation_fn = None)
    
    return fc


def fire_module(input, fire_id, channel, s1, e1, e3,):

    fire_weights = {'conv_s_1': tf.Variable(tf.truncated_normal([1, 1, channel, s1])),
                    'conv_e_1': tf.Variable(tf.truncated_normal([1, 1, s1, e1])),
                    'conv_e_3': tf.Variable(tf.truncated_normal([3, 3, s1, e3]))}

    fire_biases = {'conv_s_1': tf.Variable(tf.truncated_normal([s1])),
                   'conv_e_1': tf.Variable(tf.truncated_normal([e1])),
                   'conv_e_3': tf.Variable(tf.truncated_normal([e3]))}

    with tf.name_scope(fire_id):
        output = tf.nn.conv2d(input, fire_weights['conv_s_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_s_1')
        output = tf.nn.relu(tf.nn.bias_add(output, fire_biases['conv_s_1']))

        expand1 = tf.nn.conv2d(output, fire_weights['conv_e_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_1')
        expand1 = tf.nn.bias_add(expand1, fire_biases['conv_e_1'])

        expand3 = tf.nn.conv2d(output, fire_weights['conv_e_3'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_3')
        expand3 = tf.nn.bias_add(expand3, fire_biases['conv_e_3'])

        result = tf.concat([expand1, expand3], 3, name='concat_e1_e3')
        return tf.nn.relu(result)


def squeeze_net(input, classes):

    weights = {'conv1': tf.Variable(tf.truncated_normal([7, 7, 1, 96])),
               'conv10': tf.Variable(tf.truncated_normal([1, 1, 512, classes]))}

    biases = {'conv1': tf.Variable(tf.truncated_normal([96])),
              'conv10': tf.Variable(tf.truncated_normal([classes]))}

    output = tf.nn.conv2d(input, weights['conv1'], strides=[1,2,2,1], padding='SAME', name='conv1')
    output = tf.nn.bias_add(output, biases['conv1'])

    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')

    output = fire_module(output, s1=16, e1=64, e3=64, channel=96, fire_id='fire2')
    output = fire_module(output, s1=16, e1=64, e3=64, channel=128, fire_id='fire3')
    output = fire_module(output, s1=32, e1=128, e3=128, channel=128, fire_id='fire4')

    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')

    output = fire_module(output, s1=32, e1=128, e3=128, channel=256, fire_id='fire5')
    output = fire_module(output, s1=48, e1=192, e3=192, channel=256, fire_id='fire6')
    output = fire_module(output, s1=48, e1=192, e3=192, channel=384, fire_id='fire7')
    output = fire_module(output, s1=64, e1=256, e3=256, channel=384, fire_id='fire8')

    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool8')

    output = fire_module(output, s1=64, e1=256, e3=256, channel=512, fire_id='fire9')

    output = tf.nn.dropout(output, keep_prob=0.5, name='dropout9')

    output = tf.nn.conv2d(output, weights['conv10'], strides=[1, 1, 1, 1], padding='SAME', name='conv10')
    output = tf.nn.bias_add(output, biases['conv10'])

    output = tf.nn.avg_pool(output, ksize=[1, 13, 13, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool10')
    #output = tf.layers.conv2d(output, num_classes, 1, 1, padding = 'valid')
    P_fl = tf.contrib.layers.flatten(output)
    fc = tf.contrib.layers.fully_connected(P_fl, num_classes, activation_fn = None)
    
    print('output shape', fc.shape)
    return fc


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


minibatch_size = 256
learning_rate = 1e-3
num_epochs = 1001
seed = 10
                           

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
#Z4 = squeeze_net(Z3, num_classes)

cost1 = tf.losses.mean_squared_error(Z3, Y_img)
cost1 = tf.reduce_mean(cost1)

cost2 = tf.nn.softmax_cross_entropy_with_logits(logits = Z4, labels = Y)
cost2 = tf.reduce_mean(cost2)

cost = 0.01*cost1 + cost2



optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)    
init = tf.global_variables_initializer()    
saver = tf.train.Saver()

with tf.Session() as sess:
        
    # Run the initialization
    sess.run(init)

    x=0
    for v in tf.trainable_variables():
        x+=np.prod(v.get_shape().as_list())
        print('Trainable Parameters:',v, np.prod(v.get_shape().as_list()),"trainable parameters: ",x)

    for epoch in range(num_epochs):

        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_image, Y_train, minibatch_size, seed)
        temp_cost1 = 0
        temp_cost = 0
        for minibatch in minibatches:

            (minibatch_X, minibatch_Yi, minibatch_Y) = minibatch
            _ , temp_cost, temp_cost1, temp_cost2 = sess.run([optimizer, cost, cost1, cost2], feed_dict = {X: minibatch_X, Y_img: minibatch_Yi, Y: minibatch_Y})
            #_ , temp_cost, temp_cost1, Z31 = sess.run([optimizer, cost, cost1, Z3[0]], feed_dict = {X: minibatch_X, Y_img: minibatch_Yi})

            temp_cost += temp_cost
            temp_cost1 += temp_cost1
            temp_cost2 += temp_cost2
        #temp_cost1 = temp_cost1/0.003
        psnr = 10*np.log10(255*255/(temp_cost1))   

        # Print the cost every epoch
        if epoch % 1 == 0:
            t2 = time.time()
            #print(np.linalg.norm(Z31))
            print ("Epoch:", epoch, 'Time:', round(t2-t1, 1), 'Total loss:', round(temp_cost, 6), 'FR Loss:', round(temp_cost2, 6), 'SR Loss:', round(temp_cost1, 6), 'PSNR:', round(psnr, 6))
            #print(np.max(Z31), np.min(Z31))
            #plt.imsave('/media/ml/Data Disk/upscaling/output_images/' + str(epoch) + '.jpg', Z31)
            t1 = time.time()
            costs.append(temp_cost)
            #img = Z4.eval()
            #plt.imshow(Z31)
            #plt.show()
        if epoch % 10 == 0:  
            saver.save(sess, '/media/dl/DL/aashish/upscaling/model/model.ckpt')
            #file = open('/media/ml/Data Disk/upscaling/model/cost.txt', 'w')
            #file.write(str(cost))
            #file.close()
            print('Model Saved...')


    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    predict_op = tf.argmax(Z3, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
    #, X_t: X_tiny, Y_t:Y_tiny
    print("Train Accuracy:", train_accuracy)
