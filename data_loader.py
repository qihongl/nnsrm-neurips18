from keras.datasets import cifar100, cifar10, mnist
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
import os 

# load data
def load_cifar(data_name):
    """
    useage: 
    X_train, X_test, Y_train, Y_test, y_train, y_test, data_info = load_cifar(data_name)
    [n_classes, img_rows, img_cols, img_channels] = data_info
    """
    if data_name == 'cifar100':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        n_classes = 100
    elif data_name == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        n_classes = 10
    else:
        raise ValueError()
    # CIFAR specs
    img_rows, img_cols = 32, 32
    img_channels = 3
    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # subtract mean and normalize
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train /= 128.
    X_test /= 128.
    # data info 
    data_info = [n_classes, img_rows, img_cols, img_channels]
    return X_train, X_test, Y_train, Y_test, y_train, y_test, data_info

def load_mnist(shape = 'std'):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # data info 
    img_rows, img_cols = 28, 28
    img_channels = 1
    num_classes = 10
    data_info = [num_classes, img_rows, img_cols, img_channels]
    
    # preproc
    if shape == 'std': 
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    elif shape == 'conv':
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
        # input_shape = (img_rows, img_cols, img_channels)
    else:
        raise ValueError()        
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)
    return x_train, x_test, Y_train, Y_test, y_train, y_test, data_info


def load_imdb(max_features = 20000, maxlen = 80):
    from keras.datasets import imdb
    from keras.preprocessing import sequence
    # load 
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return x_train, y_train, x_test, y_test

