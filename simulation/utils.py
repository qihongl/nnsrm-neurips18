import numpy as np
from scipy.spatial import procrustes
from scipy.stats.stats import pearsonr, kendalltau
import seaborn as sns


def random_ortho_transform(X):
    # transform the input matrix by
    # left multiplying some random orthogonal matrix
    #
    # assume x in n by m
    n, _ = np.shape(X)
    # QR decompose a random uniform matrix
    rnd_matrix = np.random.random_sample((n, n))
    O, _ = np.linalg.qr(rnd_matrix)
    transformed_X = O @ X
    return transformed_X, O


def get_rand_ortho_matrix(n):
    # QR decompose a random uniform matrix
    rnd_matrix = np.random.random_sample((n, n))
    O, _ = np.linalg.qr(rnd_matrix)
    return O


def correlate_solution_to_truth(RSM1, RSM2, RSM_truth):
    """ compute the linear/rank correlation
        between intersubject RSM to within subject SRM
    """
    # in the shared space
    r1, _ = pearsonr(np.reshape(RSM1, [-1, ]), np.reshape(RSM_truth, [-1, ]))
    tau1, _ = kendalltau(np.reshape(RSM1, [-1, ]), np.reshape(RSM_truth, [-1, ]))
    # in the native space
    r2, _ = pearsonr(np.reshape(RSM2, [-1, ]), np.reshape(RSM_truth, [-1, ]))
    tau2, _ = pearsonr(np.reshape(RSM2, [-1, ]), np.reshape(RSM_truth, [-1, ]))
    return r1, tau1, r2, tau2


def compute_procrustes_dist_mat(matrix_array):
    # input: matrix_array, n_subj x n_units x n_examples
    n_nets = np.shape(matrix_array)[0]
    D = np.zeros((n_nets, n_nets))
    for i in range(n_nets):
        for j in np.arange(0, i):
            _, _, D[i, j] = procrustes(matrix_array[i], matrix_array[j])
    return D


def get_random_noise(size, var):
    noise_matrix = np.random.normal(size=size) * var
    return noise_matrix


def enlarge_null_space(A):
    m, n = np.shape(A)
    rank_bound = np.min([m, n])
    U, s_vals, VT = np.linalg.svd(A)
    s_vals[:int(rank_bound/2)] = 0
    A = U @ np.diag(s_vals) @ VT
    return A


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font='serif', font_scale=1.3)
    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
