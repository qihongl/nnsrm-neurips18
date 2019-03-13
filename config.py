import numpy as np 

"""
constants
"""

CIFAR10_ORDER = [2, 3, 4, 5, 6, 7, 0, 8, 1, 9]


"""
helper funcs
"""

def get_log_info(data_name, model_name):
    """Get log info
    i) which layer/epochs are saved
    ii) how many subjects do we have
    
    Parameters
    ----------
    data_name [str]: in ['cifar10','cifar100','mnist']
    model_name [str]: in ['std','conv','resnet']
    
    Returns
    ----------
    log_info [a list of stuff]
    """
    if 'resnet' in model_name: 
        # define params 
        n_subjs = 8
        log_epochs = np.round(np.logspace(0,2,10))
        log_epochs = np.insert(log_epochs, 0, 0).astype(int)
        layer_selected = [2, 10, 17, 25, 32, 40, 47, 55, 62]
    elif data_name == 'mnist' and model_name == 'std':
        n_subjs = 10
        log_epochs = np.round(np.logspace(0, 1.3, 10))
        log_epochs[0] = 0 
        layer_selected = [0,1,2]
    elif model_name == 'conv' and 'cifar' in data_name: 
        n_subjs = 10
        log_epochs = np.unique(np.round(np.logspace(0,1.7,10)))
        log_epochs = np.array([0] + list(log_epochs))
        layer_selected = [4,10,13,16]
    else: 
        raise ValueError('unrecog net class')        
    # compute some stats 
    n_log_epochs = len(log_epochs)
    n_layers = len(layer_selected)    
    return n_subjs, layer_selected, n_layers, log_epochs, n_log_epochs


def get_layer_names_plt(model_name): 
    """more interpretable layer name for plotting 
    """
    if 'resnet' in model_name: 
        layer_names = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    elif model_name == 'conv': 
        layer_names = [1, 2, 3, 4]
    else:
        raise ValueError('unrecog net class')
    return layer_names


def get_save_layer_ids(model, model_name):
    """Get ids for layers that we'd like to save 
    Parameters
    ----------
    model [keras model]
    model_name [str]
    
    Returns
    ----------
    final_sel_layer_ids [list]: layer ids 
    """
    from keras.layers import BatchNormalization, MaxPool2D, Dense
    if model_name == 'resnet18':
        # get targeted layers 
        layer_type = type(BatchNormalization())
        sel_layer_ids = []
        for l in range(len(model.layers)): 
            if type(model.layers[l]) == layer_type: 
                sel_layer_ids.append(l)
        # get even layers only, and the last dense layer
        final_sel_layer_ids = sel_layer_ids[::2] + [len(acts)-1]
    elif model_name == 'conv': 
        final_sel_layer_ids = []
        layer_types = [type(MaxPool2D()), type(Dense(0))]
        for l in range(len(model.layers)): 
            if type(model.layers[l]) in layer_types: 
                final_sel_layer_ids.append(l)
    else:
        raise ('unrecog model')    
    return final_sel_layer_ids

