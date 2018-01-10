#There is an important issue in the original code posted here: https://github.com/GPflow/GPflow/issues/505
#The problem with the above code is that the total memory usage keep increasing over time.
#I fixed the problem by saving and loading model variables before training the model with only one epoch
#It is not nice, but it is the simplest solution, perhaps.
#Script to run the code:
#max=30
#for i in `seq 1 $max`
#do
#  python -u code_just_load_dev.py $i
#done
#So I can have the results with DKL
#Specificially, with 25 iterations:
#Baseline: Epoch:  17
#Valid:
#F1-BAD:  0.478805006056 F1-OK:  0.893875873407
#F1-score multiplied:  0.42799224298
#Test:
#F1-BAD:  0.447785453068 F1-OK:  0.888451120414
#F1-score multiplied:  0.397835487483  
#RBF: Best Dev:
#Epoch:  16
#Result from the previous epoch:
#F1-BAD:  0.478700043917 F1-OK:  0.904003234937
#F1-score multiplied:  0.432746388266
#Result from the previous epoch:
#F1-BAD:  0.435914405471 F1-OK:  0.896707735811
#F1-score multiplied:  0.390887819537
#The data (RBF) can be found here:
#https://drive.google.com/file/d/1EJk1a4uH9y8yKcu2cwQNn2a-yzszOhsQ/view?usp=sharing

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
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import sys


class DataPlaceholders(object):
    def __init__(self):
        self.data = tf.placeholder(tf.float32)        
        self.keep_prob = tf.placeholder(tf.float32)        
        self.label = tf.placeholder(tf.int32, shape=[None, 1], name="labels")



def create_bias(shape, initial_val=0.1, dtype=tf.float32):
    initial = tf.constant(initial_val, shape=shape, dtype=dtype, name="bias")
    return initial


def make_feedforward_nn(x_placeholder, keep_prob_placeholder, end_h=50):
    
    with tf.name_scope("small_convnet"):
        with tf.name_scope("layer1"):
            W1 = tf.get_variable("W1", shape=[72, 512], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("b1", initializer=create_bias([512]))
            h1 = tf.nn.relu(tf.matmul(x_placeholder, W1) + b1)
        with tf.name_scope("layer2"):
            W2 = tf.get_variable("W2", shape=[512, end_h], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2", initializer=create_bias([end_h]))
            h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
            #h2 = (tf.matmul(h1, W2) + b2)
            #h4_drop = tf.nn.dropout(h4, keep_prob_placeholder)
    return h2


def suggest_good_intial_inducing_points(phs: DataPlaceholders, x_data, h, tf_session, num_inducing):
    h_data = tf_session.run(h, feed_dict={phs.data: x_data, phs.keep_prob: 1.0})
    kmeans = cluster.MiniBatchKMeans(n_clusters=num_inducing, batch_size=num_inducing*10)
    kmeans.fit(h_data)
    new_inducing = kmeans.cluster_centers_
    return new_inducing


def suggest_sensible_lengthscale(phs: DataPlaceholders, x_data, h, tf_session):
    h_data = tf_session.run(h, feed_dict={phs.data: x_data, phs.keep_prob: 1.0})
    lengthscale = np.mean(distance.pdist(h_data, 'euclidean'))
    return lengthscale



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



def main(number_epoch):
    
    """
    Simple demonstration of how you can put a GP on top of a NN and train the whole system end-to-end in GPflow-1.0.


    Note
    that in the new GPflow there are new features that we do not take advantage of here but could be used to make
    the whole example cleaner. For example you may want to use a gpflow.train Optimiser as this will take care of
    passing in the GP model feed dict for you as well as initially initialising the optimisers TF variables.
    You could also choose to tell the gpmodel to initialise the NN variables by subclassing SVGP and overriding the
    appropriate variable initialisation method.
    """
    # ## We load in the MNIST data. We will create a validation set but will not use it in this simple example.
    
    dataset = np.loadtxt("test", delimiter=",")
    x_test = dataset[:,0:72]
    y_test = dataset[:,72].reshape(-1,1)
    #print(x_test[20])

    dataset = np.loadtxt("dev", delimiter=",")
    x_valid = dataset[:,0:72]
    y_valid = dataset[:,72].reshape(-1,1)
    #resampleFile()
    dataset = np.loadtxt("train.revised", delimiter=",")
    x_train = dataset[:,0:72]
    y_train = dataset[:,72].reshape(-1,1)

    
    x_train_root = x_train
    x_valid_root = x_valid
    x_train, x_test, x_valid = standardize_data(copy.deepcopy(x_train_root), x_test, copy.deepcopy(x_valid_root))

        # ## We set up a TensorFlow Graph and a Session linked to this.
    tf_graph = tf.Graph()
    tf_session = tf.Session(graph=tf_graph)

    # ## We have some settings for the model and its training which we will set up below.
    num_h = 17
    num_classes = 2 #could be improved here
    num_inducing = 100
    minibatch_size = 250

    # ## We set up the NN part of the GP kernel. This needs to be put on the same graph
    with tf_graph.as_default():
        phs = DataPlaceholders()
        nn_base = tf.make_template("sconvnet_kernel", make_feedforward_nn, end_h=num_h)  # end h is the number of hidden
        # units at the end

        h = nn_base(phs.data, phs.keep_prob)
        h = tf.cast(h, gpf.settings.tf_float)

        nn_vars = tf.global_variables()  # only nn variables exist up to now.
    tf_session.run(tf.variables_initializer(nn_vars))



    # ## We now set up the GP part. Instead of the usual X data it will get the data after being processed by the NN.
    with gpf.defer_build():
        kernel = gpf.kernels.RBF(num_h)
        likelihood = gpf.likelihoods.MultiClass(num_classes)
        gp_model = gpf.models.SVGP(h, phs.label, kernel, likelihood, np.ones((num_inducing, num_h), gpf.settings.np_float),
                               num_latent=num_classes, whiten=False, minibatch_size=None, num_data=x_train.shape[0])
    # ^ so we say minibatch size is None to make sure we get DataHolder rather than minibatch data holder, which
    # does not allow us to give in tensors. But we will handle all our minibatching outside.
    gp_model.compile(tf_session)

    # ## The initial lengthscales and inducing point locations are likely very bad. So we use heuristics for good
    # initial starting points and reset them at these values.

    gp_model.Z.assign(suggest_good_intial_inducing_points(phs, np.unique(x_train, axis=0)[:5000, :], 
        h, tf_session, num_inducing), tf_session)
    gp_model.kern.lengthscales.assign(suggest_sensible_lengthscale(phs, np.unique(x_train, axis=0)[:5000, :], h, tf_session) 
        + np.zeros_like(gp_model.kern.lengthscales.read_value()), tf_session)

   

    # ^ note that this assign should reapply the transform for us :). The zeros ND array exists to make sure
    # the lengthscales are the correct shape via  broadcasting

    # ## We create ops to measure the predictive log likelihood and the accuracy.
    with tf_graph.as_default():
        log_likelihood_predict = gp_model.likelihood.predict_density(*gp_model._build_predict(h), phs.label)
        outputs = tf.argmax(gp_model.likelihood.predict_mean_and_var(*gp_model._build_predict(h))[0], axis=1, output_type=tf.int32)
        accuracy = tf.cast(tf.equal(outputs, tf.squeeze(phs.label)), tf.float32)
        avg_acc = tf.reduce_mean(accuracy)
        avg_ll = tf.reduce_mean(log_likelihood_predict)

        # ## we now create an optimiser and initialise its variables. Note that you could use a GPflow optimiser here
        # and this would now be done for you.
        all_vars_up_to_trainer = tf.global_variables()
        optimiser = tf.train.AdamOptimizer(1e-4)
        print(tf.global_variables())
        minimise = optimiser.minimize(gp_model.objective)  # this should pick up all Trainable variables.
        adam_vars = list(set(tf.global_variables()) - set(all_vars_up_to_trainer))
        tf_session.run(tf.variables_initializer(adam_vars))
        saver = tf.train.Saver()



    # ## We now go through a training loop where we optimise the NN and GP. we will print out the test results at
    # regular intervals.    
    results = []
    np.random.seed(number_epoch) #I did this to make sure I have a different randomization
    
    
    print("Epoch: ", number_epoch)
    i = number_epoch
    if 1==1:
    #for i in range(2): #100 epochs
        if i>0:
            #load truoc
            #print_tensors_in_checkpoint_file(file_name="model_at_epoch"+str(i-1)+".ckpt", tensor_name='', all_tensors=False)

            saver.restore(tf_session, "model_at_epoch"+str(i-1)+".ckpt")

            fd = gp_model.feeds or {}
            fd.update({phs.keep_prob: 1.0, phs.data: x_test,
                       phs.label: y_test})
            accuracy_evald, log_like_evald, outputs_evald = tf_session.run([avg_acc, avg_ll, outputs], feed_dict=fd)
            #print("Epoch {}: Loss is {}. \nTest set LL {}, Acc {}, Outputs {}".format(i, loss_evd, log_like_evald,
            #                                                                  accuracy_evald, outputs_evald))
            outputs_evald = outputs_evald.reshape(-1, 1)
            #print(outputs_evald)
            print("Result from the previous epoch:")
            compute_scores(y_test, outputs_evald)



        shuffle = np.arange(y_train.size)
        np.random.shuffle(shuffle)
        print(shuffle)
        x_train_shuffle = x_train[shuffle]
        y_train_shuffle = y_train[shuffle]
        data_indx = 0
        while data_indx<y_train.size:
            lastIndex = data_indx + minibatch_size
            if lastIndex>=y_train.size:
                lastIndex = y_train.size
            indx_array = np.mod(np.arange(data_indx, lastIndex), x_train_shuffle.shape[0])
            #print("array", indx_array)
            data_indx += minibatch_size
            #print(data_indx)fz
            fd = gp_model.feeds or {}
            fd.update({
                phs.keep_prob: 1.0,
                phs.data: x_train_shuffle[indx_array],
                phs.label: y_train_shuffle[indx_array]
                })
            _, loss_evd = tf_session.run([minimise, -gp_model.objective], feed_dict=fd)            
            # Print progress every 1 epoch.
        save_path = saver.save(tf_session, "./model_at_epoch"+str(i)+".ckpt")        

            #results.append(dict(step_no=i, loss=loss_evd, test_acc=accuracy_evald, test_ll=log_like_evald))

    print("Done!")



if __name__ == '__main__':
    #for arg in sys.argv[1:]:
    #    print(arg)
    main(int(sys.argv[1:][0]))
