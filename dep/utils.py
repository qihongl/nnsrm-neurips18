import numpy as np 
from qmvpa import utils

def subset_units(acts_matrix, n_max_units):
    n_units = np.shape(acts_matrix)[1]
    assert n_units > n_max_units
    uids = np.random.choice(n_units, n_max_units, replace=False)
    return acts_matrix[:, uids]


def smooth_colwise(mat, ws):
    wmat = [utils.mov_mean(mat[:, v], ws) 
            for v in range(np.shape(mat)[1])]
    wmat = np.array(wmat).T
    return wmat