# from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Dense
from data_loader import load_cifar
from sklearn.model_selection import train_test_split
import os
import sys 
import numpy as np
import resnet
from dep.read_acts_keras import get_activations
from models import get_cifar_convnet
from config import get_log_info, get_save_layer_ids

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
log_root = '/tigress/qlu/logs/keras-resnet/log'

data_name = sys.argv[1]
# data_name = 'cifar10'
# model_name = 'resnet18'
model_name = 'conv'

n_subjs = 10

# define params 
log_info_list = get_log_info(data_name, model_name)
n_subjs, layer_selected, n_layers, log_epochs, n_log_epochs = log_info_list

print('data: %s\nmodel: %s' % (data_name, model_name))
print('log epochs', log_epochs)
print('layer selected', layer_selected)

# subj_id = 1
# target_epoch = 0

for subj_id in range(n_subjs):
    # for subj_id in [5,6,7]:
    for target_epoch in log_epochs:
        # get dirs
        log_dir = os.path.join(log_root, data_name, model_name, 'subj%.2d' % (subj_id))
        wts_path = os.path.join(log_dir, 'weights.%.3d.hdf5' % (target_epoch))
        # make activation log dir 
        acts_path = os.path.join(log_dir, 'epoch_%.3d' % target_epoch, 'activations')
        print(acts_path)
        if not os.path.exists(acts_path):
            os.makedirs(acts_path)

        # get data 
        _, X_test, _, Y_test, _, y_test, data_info = load_cifar(data_name)
        [n_classes, img_rows, img_cols, img_channels] = data_info
        n_test_egs = len(y_test)

        # get pre-trained model 
        if model_name == 'resnet18':
            model = resnet.ResnetBuilder.build_resnet_18(
                (img_channels, img_rows, img_cols), n_classes)
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam', metrics=['accuracy'])
            model.load_weights(wts_path)
        elif model_name == 'conv': 
            model = get_cifar_convnet((img_rows, img_cols, img_channels), n_classes)
            model.load_weights(wts_path)
        else:
            raise ('unrecog model')

        # get the ids for a subet of layers - to be saved  
        final_sel_layer_ids = get_save_layer_ids(model, model_name)
        print(final_sel_layer_ids)

        # get activations on the test set 
        acts = get_activations(model, X_test, print_shape_only = True);
        n_layers = len(acts)
        print('n_layers = ', n_layers)

        # save activity 
        n_final_sel_layers = len(final_sel_layer_ids)
        for l in range(n_final_sel_layers): 
            raw_layer_idx = final_sel_layer_ids[l]
            acts_path_l = os.path.join(acts_path, 'layer_%.3d' % (raw_layer_idx))
            np.save(acts_path_l, acts[raw_layer_idx])
            print(acts_path_l)    