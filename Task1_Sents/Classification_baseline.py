
import os
import copy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data
import numpy as np
from sklearn import cluster
from scipy.spatial import distance
import pandas as pd
from keras.utils import np_utils
import gpflow as gpf
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from numpy import argmax
from sklearn.metrics import mean_absolute_error

def create_bias(shape, initial_val=0.1, dtype=tf.float32):
    initial = tf.constant(initial_val, shape=shape, dtype=dtype, name="bias")
    return initial





def standardize_data(X_train, X_test, X_valid):
    unique_X_train = np.unique(X_train, axis=0)
    X_mean = np.mean(unique_X_train, axis=0)
    #print(X_mean)
    X_std = np.std(unique_X_train, axis=0)+0.0000001 #a small noise
    #print(X_std)
    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    X_valid -= X_mean
    X_valid /= X_std

    return X_train, X_test, X_valid



def standardize_data(X_train, X_test2016, X_test2017, X_valid):
    unique_X_train = np.unique(X_train, axis=0)
    X_mean = np.mean(unique_X_train, axis=0)
    #print(X_mean)
    X_std = np.std(unique_X_train, axis=0)+0.0000001 #a small noise
    #print(X_std)
    X_train -= X_mean
    X_train /= X_std
    X_test2016 -= X_mean
    X_test2016 /= X_std

    X_test2017 -= X_mean
    X_test2017 /= X_std


    X_valid -= X_mean
    X_valid /= X_std

    return X_train, X_test2016, X_test2017, X_valid


def compute_scores(flat_true, flat_pred):
    f1_bad, f1_good = f1_score(flat_true, flat_pred, average=None, pos_label=None)
    print("F1-BAD: ", f1_bad, "F1-OK: ", f1_good)
    print("F1-score multiplied: ", f1_bad * f1_good)
def getMax(x):
    output = []
    for y in x:
        output.append(argmax(y, 0))
    #print(output)
    return output

def getMaxTensor(x):
    output = []
    for y in x:
        for yy in y:
            #print(yy)
            output.append(argmax(yy, 0))
    #print(output)
    return output

def resampleFile():
    filename = open("train.revised", "w")
    file = open("train.shuffle", "r")
    for x in file:
        x = x.strip()
        filename.write(x+"\n")
        #if x.endswith(",3"):
            #filename.write(x+"\n")
            #filename.write(x+"\n")
            #filename.write(x+"\n")
            #filename.write(x+"\n")
    filename.close()
    file.close()

x = tf.placeholder("float", [None, 17])
y = tf.placeholder("float", [None, 3])

def make_feedforward_nn(x):
    W1 = tf.get_variable("W1", shape=[17, 512], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", initializer=create_bias([512]))
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    W2 = tf.get_variable("W2", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", initializer=create_bias([256]))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    W3 = tf.get_variable("W3", shape=[256, 17], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", initializer=create_bias([17]))
    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

    W4 = tf.get_variable("W4", shape=[17, 3], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", initializer=create_bias([3]))
    h4 = (tf.matmul(h3, W4) + b4)
    return h4


def main():
    dataset = np.loadtxt("test", delimiter=",")
    x_test_2016 = dataset[:,0:17]
    y_test_2016 = dataset[:,17].reshape(-1,1)
    y_test_2016 = y_test_2016-1
    y_test_2016 = to_categorical(y_test_2016)
    #print(encoded)


    dataset = np.loadtxt("test2", delimiter=",")
    x_test_2017 = dataset[:,0:17]
    y_test_2017 = dataset[:,17].reshape(-1,1)
    y_test_2017 = y_test_2017-1
    y_test_2017 = to_categorical(y_test_2017)
    #print(x_test[20])

    dataset = np.loadtxt("dev.shuffle", delimiter=",")
    x_valid = dataset[:,0:17]
    y_valid = dataset[:,17].reshape(-1,1)

    y_valid = y_valid-1
    y_valid = to_categorical(y_valid)

    resampleFile()
    dataset = np.loadtxt("train.revised", delimiter=",")
    x_train = dataset[:,0:17]
    y_train = dataset[:,17].reshape(-1,1)

    y_train = y_train-1
    y_train = to_categorical(y_train)

    
    x_train_root = x_train
    x_valid_root = x_valid
    x_train, x_test_2016, x_test_2017, x_valid = standardize_data(copy.deepcopy(x_train_root), x_test_2016, x_test_2017, copy.deepcopy(x_valid_root))

    

    # ## We have some settings for the model and its training which we will set up below.
    num_h = 17
    num_classes = 3 #could be improved here
    num_inducing = 100
    minibatch_size = len(y_train)
    #minibatch_size = 128


    print(len(y_train))
    print(len(y_test_2016))
    print(len(y_test_2017))
    print(len(y_valid))


    model = make_feedforward_nn(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    predict = tf.nn.softmax(model)
    

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep = 1000)
    dev_max = 1000
    test_max=1000
    test_adaptation_max = 1000
    output_max = []
    epoch_max = 1000


    with tf.Session() as sess:
        sess.run(init)
        # ## We now go through a training loop where we optimise the NN and GP. we will print out the test results at
        # regular intervals.    
        results = []    
        SEED = 449
        np.random.seed(SEED)
        for i in range(50): #100 epochs        
            print("epoch: ")
            print(i)
            if i>0:
                saver.restore(sess, "baselinemodel_at_epoch"+str((i-1))+".ckpt")
                predict_op = sess.run([predict], feed_dict={x: x_valid, y: y_valid})
                if mean_absolute_error(getMax(y_valid), (getMaxTensor(predict_op)))<dev_max:
                    dev_max = mean_absolute_error(getMax(y_valid), (getMaxTensor(predict_op)))
                    predict_op = sess.run([predict], feed_dict={x: x_test_2016, y: y_test_2016})
                    test_max = (mean_absolute_error(getMax(y_test_2016), getMaxTensor(predict_op)))
                    output_max = getMaxTensor(predict_op)
                    predict_op = sess.run([predict], feed_dict={x: x_test_2017, y: y_test_2017})
                    test_adaptation_max = (mean_absolute_error(getMax(y_test_2017), (getMaxTensor(predict_op))))
                    epoch_max = (i-1)

            shuffle = np.arange(len(y_train))        
            np.random.shuffle(shuffle)
            #print(shuffle)
            x_train_shuffle = x_train[shuffle]
            y_train_shuffle = y_train[shuffle]
            data_indx = 0
            while data_indx<len(y_train):
                lastIndex = data_indx + minibatch_size
                if lastIndex>=len(y_train):
                    lastIndex = len(y_train)
                indx_array = np.mod(np.arange(data_indx, lastIndex), x_train_shuffle.shape[0])
                data_indx += minibatch_size
                sess.run([optimizer,cost], feed_dict={x: x_train_shuffle[indx_array], y: y_train_shuffle[indx_array]})
            save_path = saver.save(sess, "./baselinemodel_at_epoch"+str((i))+".ckpt")

            
    print("Done!")
    print(test_max, test_adaptation_max, dev_max, output_max, epoch_max)



if __name__ == '__main__':
    main()
