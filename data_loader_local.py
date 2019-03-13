import numpy as np 
import os 

data_root = '/tigress/qlu/data/keras-nn-srm/data/'
data_format = '.npz'


def load_data(data_name):
    """
    data avail: 'cifar10', 'cifar100', 'mnist_std', 'mnist_conv'
    data_info = [num_classes, img_rows, img_cols, img_channels]
    """
    data_path = os.path.join(data_root, data_name + data_format)
    data = np.load(data_path)
    return unpack_data(data)

def unpack_data(data):
    X_train = data['X_train']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_test = data['Y_test']
    y_train = data['y_train']
    y_test = data['y_test']
    data_info = data['data_info']
    return X_train, X_test, Y_train, Y_test, y_train, y_test, data_info