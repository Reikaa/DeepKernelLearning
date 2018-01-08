#the best result on validation set will be returned



import numpy as np
import sys
np.random.seed(42)
import copy

# Keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from keras.callbacks import ModelCheckpoint
from keras import backend as K


def standardize_data(X_train, X_test, X_valid):
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)+0.0000001

    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    X_valid -= X_mean
    X_valid /= X_std

    return X_train, X_test, X_valid


def assemble_mlp(input_shape, output_shape):
    """Assemble a simple MLP model.
    """
    
    return model

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
    # Load data

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

    

    #X_train, X_test, X_valid = standardize_data(X_train, X_test, X_valid)
    data = {
        'train': (x_train, y_train),
        'valid': (x_valid, y_valid),
        'test': (x_test, y_test),
    }

    # Model & training parameters
    input_shape = data['train'][0].shape[1:]    
    output_shape = data['train'][1].shape[1:]
    batch_size = 128
    epochs = 50

    # Construct & compile the model

    model = Sequential()
    model.add(Dense(512, input_dim=72, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')
    # checkpoint

    count = 0
    while count < 50:
        filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, verbose=0, save_best_only=False, mode='auto')
        callbacks_list = [checkpoint]
        # Fit the model
        model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=5, batch_size=250, verbose=1, callbacks = callbacks_list)
        model.load_weights("weights.best.hdf5")
        x_test, y_test = data['valid']
        y_preds = model.predict(x_test)
        y_preds_binary = []
        for x in y_preds:
            if x > 0.5:
                x = 1
            else:
                x = 0
            y_preds_binary.append(x)
        print("Valid:")
        compute_scores(y_test, y_preds_binary)

        x_test, y_test = data['test']
        y_preds = model.predict(x_test)
        y_preds_binary = []
        for x in y_preds:
            if x > 0.5:
                x = 1
            else:
                x = 0
            y_preds_binary.append(x)
        print("Test:")
        compute_scores(y_test, y_preds_binary)
        count = count + 1    


if __name__ == '__main__':
    main()
