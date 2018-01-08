
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


class DataPlaceholders(object):
    def __init__(self):
        self.ximage_flat = tf.placeholder(tf.float32)        
        self.label = tf.placeholder(tf.int32, shape=[None, 1], name="labels")


def get_mnist():
    mnist = mnist_input_data.read_data_sets(os.path.join(os.path.dirname(__file__), "mnist_data/"), one_hot=False, validation_size=0)

    norm_data = lambda img_in: 2.*img_in - 1.
    add_extra_dim = lambda x: x[:, np.newaxis]

    return (norm_data(mnist.train.images), add_extra_dim(mnist.train.labels),
            norm_data(mnist.validation.images), add_extra_dim(mnist.validation.labels),
            norm_data(mnist.test.images), add_extra_dim(mnist.test.labels))


def create_weight(shape, stddev=0.1, dtype=tf.float32):
    inital = tf.truncated_normal(shape, dtype=dtype, stddev=stddev, name="weight")
    return inital


def create_bias(shape, initial_val=0.1, dtype=tf.float32):
    initial = tf.constant(initial_val, shape=shape, dtype=dtype, name="bias")
    return initial


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name="2dconv")


def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="max_poll2by2")


def make_small_mnist_nn(x_placeholder, end_h=50):
    with tf.name_scope("small_convnet"):
        with tf.name_scope("layer1"):
            W1 = tf.get_variable("W1", initializer=create_weight([72, 1024]))
            b1 = tf.get_variable("b1", initializer=create_weight([1024]))

            h1 = tf.nn.relu(tf.matmul(x_placeholder, W1) + b1)

        with tf.name_scope("layer2"):
            W2 = tf.get_variable("W2", initializer=create_weight([1024, end_h]))
            b2 = tf.get_variable("b2", initializer=create_bias([end_h]))
            h4 = tf.matmul(h1, W2) + b2

    return h4


def suggest_good_intial_inducing_points(phs: DataPlaceholders, x_data, h, tf_session, num_inducing):
    h_data = tf_session.run(h, feed_dict={phs.ximage_flat: x_data})
    kmeans = cluster.MiniBatchKMeans(n_clusters=num_inducing, batch_size=num_inducing*10)
    kmeans.fit(h_data)
    new_inducing = kmeans.cluster_centers_
    return new_inducing


def suggest_sensible_lengthscale(phs: DataPlaceholders, x_data, h, tf_session):
    h_data = tf_session.run(h, feed_dict={phs.ximage_flat: x_data})
    lengthscale = np.mean(distance.pdist(h_data, 'euclidean'))
    return lengthscale


def standardize_data(X_train, X_test, X_valid):
    unique_X_train = np.unique(X_train, axis=0)
    X_mean = np.mean(unique_X_train, axis=0, dtype=np.float64)
    #print(X_mean)
    X_std = np.std(unique_X_train, axis=0, dtype=np.float64)+0.0000000001 #a small noise
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

def main():
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
    
    dataset = np.loadtxt("test", delimiter=",", dtype = np.float64)
    x_test = dataset[:,0:72]
    y_test = dataset[:,72].reshape(-1,1)
    #print(x_test[20])

    dataset = np.loadtxt("dev", delimiter=",", dtype = np.float64)
    x_valid = dataset[:,0:72]
    y_valid = dataset[:,72].reshape(-1,1)
    #resampleFile()
    dataset = np.loadtxt("train.revised", delimiter=",", dtype = np.float64)
    x_train = dataset[:,0:72]
    y_train = dataset[:,72].reshape(-1,1)

    
    x_train_root = x_train
    x_valid_root = x_valid
    x_train, x_test, x_valid = standardize_data(copy.deepcopy(x_train_root), x_test, copy.deepcopy(x_valid_root))

            

    rng = np.random.RandomState(100)
    train_permute = rng.permutation(x_train.shape[0])
    x_train, y_train = x_train[train_permute, :], y_train[train_permute, :]

    # ## We set up a TensorFlow Graph and a Session linked to this.
    tf_graph = tf.Graph()
    tf_session = tf.Session(graph=tf_graph)

    # ## We have some settings for the model and its training which we will set up below.
    num_h = 17
    num_classes = 2 #could be improved here
    num_inducing = 1000
    minibatch_size = 250

    # ## We set up the NN part of the GP kernel. This needs to be put on the same graph
    with tf_graph.as_default():
        phs = DataPlaceholders()
        nn_base = tf.make_template("sconvnet_kernel", make_small_mnist_nn, end_h=num_h)  # end h is the number of hidden
        # units at the end

        h = nn_base(phs.ximage_flat)
        h = tf.cast(h, gpf.settings.tf_float)

        nn_vars = tf.global_variables()  # only nn variables exist up to now.
    tf_session.run(tf.variables_initializer(nn_vars))

    # ## We now set up the GP part. Instead of the usual X data it will get the data after being processed by the NN.
    kernel = gpf.kernels.RBF(num_h)
    likelihood = gpf.likelihoods.MultiClass(num_classes)
    gp_model = gpf.models.SVGP(h, phs.label, kernel, likelihood, np.ones((num_inducing, num_h), gpf.settings.np_float),
                               num_latent=num_classes, whiten=False, minibatch_size=None, num_data=x_train.shape[0])
    # ^ so we say minibatch size is None to make sure we get DataHolder rather than minibatch data holder, which
    # does not allow us to give in tensors. But we will handle all our minibatching outside.
    gp_model.compile(tf_session)

    # ## The initial lengthscales and inducing point locations are likely very bad. So we use heuristics for good
    # initial starting points and reset them at these values.
    #gp_model.Z.assign(suggest_good_intial_inducing_points(phs, x_train[:5000, :], h, tf_session, num_inducing))
    #gp_model.kern.lengthscales.assign(suggest_sensible_lengthscale(phs, x_train[:5000, :], h, tf_session) + np.zeros_like(gp_model.kern.lengthscales.read_value()))

    gp_model.Z.assign(suggest_good_intial_inducing_points(phs, np.unique(x_train, axis=0)[:5000, :], h, tf_session, num_inducing))
    gp_model.kern.lengthscales.assign(suggest_sensible_lengthscale(phs, np.unique(x_train, axis=0)[:5000, :], h, tf_session) 
        + np.zeros_like(gp_model.kern.lengthscales.read_value()))
    # ^ note that this assign should reapply the transform for us :). The zeros ND array exists to make sure
    # the lengthscales are the correct shape via  broadcasting

    # ## We create ops to measure the predictive log likelihood and the accuracy.
    with tf_graph.as_default():
        log_likelihood_predict = gp_model.likelihood.predict_density(*gp_model._build_predict(h), phs.label)
        outputs = tf.argmax(gp_model.likelihood.predict_mean_and_var(*gp_model._build_predict(h))[0], axis=1, output_type=tf.int32)
        accuracy = tf.cast(tf.equal(tf.argmax(gp_model.likelihood.predict_mean_and_var(*gp_model._build_predict(h))[0], axis=1, output_type=tf.int32),
                                         tf.squeeze(phs.label)), tf.float32)
        avg_acc = tf.reduce_mean(accuracy)
        avg_ll = tf.reduce_mean(log_likelihood_predict)

        # ## we now create an optimiser and initialise its variables. Note that you could use a GPflow optimiser here
        # and this would now be done for you.
        all_vars_up_to_trainer = tf.global_variables()
        optimiser = tf.train.AdamOptimizer()
        print(tf.global_variables())
        minimise = optimiser.minimize(gp_model.objective)  # this should pick up all Trainable variables.
        adam_vars = list(set(tf.global_variables()) - set(all_vars_up_to_trainer))
        tf_session.run(tf.variables_initializer(adam_vars))



    # ## We now go through a training loop where we optimise the NN and GP. we will print out the test results at
    # regular intervals.
    data_indx = 0
    results = []
    print("starting")
    for i in range(6000):
        indx_array = np.mod(np.arange(data_indx, data_indx + minibatch_size), x_train.shape[0])
        data_indx += minibatch_size

        fd = gp_model.feeds or {}
        fd.update({
            phs.ximage_flat: x_train[indx_array],
            phs.label: y_train[indx_array]
        })
        _, loss_evd = tf_session.run([minimise, -gp_model.objective], feed_dict=fd)

        # Print progress every 50 steps.
        if i % 50 == 0:
            fd = gp_model.feeds or {}
            fd.update({phs.ximage_flat: x_test,
                       phs.label: y_test})
            accuracy_evald, log_like_evald, outputs_evald = tf_session.run([avg_acc, avg_ll, outputs], feed_dict=fd)
            print("Iteration {}: Loss is {}. \nTest set LL {}, Acc {}, Outputs {}".format(i, loss_evd, log_like_evald,
                                                                              accuracy_evald, outputs_evald))
            outputs_evald = outputs_evald.reshape(-1, 1)
            #print(outputs_evald)
            compute_scores(y_test, outputs_evald)

            results.append(dict(step_no=i, loss=loss_evd, test_acc=accuracy_evald, test_ll=log_like_evald))
    print("Done!")

    # ## We save the results for looking at them later.
    df = pd.DataFrame(results)
    df.to_pickle("gpdnn_mnist_train.pdpick.pdpick")


if __name__ == '__main__':
    main()
    