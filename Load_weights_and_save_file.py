
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

def compute_scores(flat_true, flat_pred):
    f1_bad, f1_good = f1_score(flat_true, flat_pred, average=None, pos_label=None)
    print("F1-BAD: ", f1_bad, "F1-OK: ", f1_good)
    print("F1-score multiplied: ", f1_bad * f1_good)

def resampleFile():
    filename = open("train.revised", "w")
    file = open("train", "r")
    for x in file:
        x = x.strip()
        filename.write(x+"\n")
        if x.endswith(",0"):
            #filename.write(x+"\n")
            filename.write(x+"\n")
    filename.close()
    file.close()


def make_feedforward_nn(x):
    W1 = tf.get_variable("W1", shape=[144, 512], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", initializer=create_bias([512]))
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    W2 = tf.get_variable("W2", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", initializer=create_bias([256]))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    W3 = tf.get_variable("W3", shape=[256, 17], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", initializer=create_bias([17]))
    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

    W4 = tf.get_variable("W4", shape=[17, 1], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", initializer=create_bias([1]))
    h4 = (tf.matmul(h3, W4) + b4)
    return h4

def convertContinuoustoOutput(y_preds):
    flat_list = []
    for sublist in y_preds:
        for item in sublist:
            flat_list.append(item)

    y_preds_binary = []
    for x in flat_list:
        if x > 0.5:
            x = 1
        else:
            x = 0
        y_preds_binary.append(x)
    return y_preds_binary

def load(epoch_number):

    x = tf.placeholder("float", [None, 144])

    y = tf.placeholder("float", [None, 1])


    model = make_feedforward_nn(x)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    predict = tf.sigmoid(model)
    

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep = 1000)

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "baselinemodel_at_epoch"+str(epoch_number-1)+".ckpt")
        variables_names =[v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k,v in zip(variables_names, values):
            print(k, v)
        return variables_names, values


if __name__ == '__main__':
    load(10)
