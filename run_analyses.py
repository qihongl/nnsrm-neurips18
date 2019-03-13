from qmvpa import utils, factor, rsa
from data_loader import load_cifar, load_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from dep.utils import subset_units
from config import get_log_info

sns.set(style = 'white', context='paper', rc={"lines.linewidth": 2.5})
SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 15
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['figure.figsize'] = 8, 8

# get roots
log_root = '/tigress/qlu/logs/keras-resnet/log'
plt_root = '/tigress/qlu/logs/keras-resnet/plots'
pltdata_root = '/tigress/qlu/logs/keras-resnet/plots_data'

# define data name, model name
data_name = sys.argv[1]
# data_name = 'cifar10'
model_name = 'conv'

# load data
# _, _, _, _, _, labels, data_info = load_mnist(data_name + '_' + model_name)
_, _, _, _, _, labels, data_info = load_cifar(data_name)
[n_classes, img_rows, img_cols, img_channels] = data_info
n_test_egs = len(labels)

# define params
log_info_list = get_log_info(data_name, model_name)
n_subjs, layer_selected, n_layers, log_epochs, n_log_epochs = log_info_list

print('data: %s\nmodel: %s' % (data_name, model_name))
print('log epochs', log_epochs)
print('layer selected', layer_selected)

# set relevant parameters
n_max_units = 1000
test_size = .2
ss = StandardScaler()

# choose layer and epoch
# e = 21
# l = 13
for e in log_epochs:
    for l in layer_selected:
        print('Layer %d, Epoch %d' % (l, e))
        # load data for all subjects
        Xs_train = []
        Xs_test = []
        for subj_id in range(n_subjs):
            # activation log dir
            log_dir = os.path.join(log_root, data_name, model_name, 'subj%.2d' % (subj_id))
            acts_path = os.path.join(log_dir, 'epoch_%.3d' % e, 'activations')
            acts_path_l = os.path.join(acts_path, 'layer_%.3d.npy' % (l))

            # load activity
            loaded_acts = np.load(acts_path_l)
            loaded_acts = np.reshape(loaded_acts, [n_test_egs, -1])

            # subset units for computational efficiency
            n_units = np.shape(loaded_acts)[1]
            if n_units > n_max_units:
                loaded_acts = subset_units(loaded_acts, n_max_units)

            # split to training and testing set
            X = loaded_acts
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size = test_size, stratify = labels, random_state=0
            )
            y_test_id = np.argsort(np.ravel(y_test))
            y_test = y_test[y_test_id]
            X_test = X_test[y_test_id,: ]
            
            # normalize the data 
            X_train = ss.fit_transform(X_train.T).T
            X_test = ss.fit_transform(X_test.T).T            
            
            # gather data
            Xs_train.append(X_train.T)
            Xs_test.append(X_test.T)

        print(len(Xs_train))
        print(np.shape(X_train))
        print(np.shape(X_test))

        # plot dir and plot data dir
        plt_dir = os.path.join(
            plt_root, data_name, model_name,
            'epoch_%.3d' % e, 'rsa'
        )
        pltdata_dir = os.path.join(
            pltdata_root, data_name, model_name,
            'epoch_%.3d' % e, 'rsa'
        )
        # make dir if do not exist
        if not os.path.exists(plt_dir):
            os.makedirs(plt_dir)
        if not os.path.exists(pltdata_dir):
            os.makedirs(pltdata_dir)
        # print out info
        print(plt_dir)
        print(pltdata_dir)

        """SRM"""
        Xs_train_s, Xs_test_s, srm, var_exp_train = factor.fit_srm(
            Xs_train, Xs_test, np.shape(Xs_train)[1])
        var_exp_test = factor.calc_srm_var_exp(Xs_test, Xs_test_s, srm.w_)
        print('var exp train: ', var_exp_train)
        print('var exp test : ', var_exp_test)

        # save SRM transformed test set activities
        srm_result = [Xs_test, Xs_test_s, y_test, srm.w_, srm.s_, srm.features, var_exp_train, var_exp_test]
        srm_result_fname = 'srm_l%.2d' % (l)
        srm_result_path = os.path.join(pltdata_dir, srm_result_fname)
        print('save SRM results to: ', srm_result_path)
        np.save(srm_result_path, srm_result)


        """RSA"""
        # compute the RDM in the native space
        wRSMs_n = rsa.within_RSMs(Xs_test)
        wRSM_n = np.mean(wRSMs_n, axis = 0)
        iRSM_n = rsa.inter_RSMs(Xs_test)
        # compute the RDM in the shared space
        wRSMs_s = rsa.within_RSMs(Xs_test_s)
        wRSM_s = np.mean(wRSMs_s, axis = 0)
        iRSM_s = rsa.inter_RSMs(Xs_test_s)

        # save SRM RSA results
        rsa_result = [wRSM_n, iRSM_n, wRSM_s, iRSM_s]
        rsa_result_fname = 'wirsa_ns_l%.2d' % (l)
        rsa_result_path = os.path.join(pltdata_dir, rsa_result_fname)
        print(rsa_result_path)
        np.save(rsa_result_path, rsa_result)


        # plots
        val_min = -1
        val_max = 1

        # compare 4 RDMs
        f, axes = plt.subplots(2,2)
        sns.heatmap(wRSM_n,
                    vmin=val_min, vmax=val_max, cbar=False,
                    xticklabels= False, yticklabels=False,
                    square = True, cmap='viridis', ax = axes[0,0])
        sns.heatmap(iRSM_n,
                    vmin=val_min, vmax=val_max, cbar=False,
                    xticklabels= False, yticklabels=False,
                    square = True, cmap='viridis', ax = axes[0,1])
        sns.heatmap(wRSM_s,
                    vmin=val_min, vmax=val_max, cbar=False,
                    xticklabels= False, yticklabels=False,
                    square = True, cmap='viridis', ax = axes[1,0])
        sns.heatmap(iRSM_s,
                    vmin=val_min, vmax=val_max, cbar=False,
                    xticklabels= False, yticklabels=False,
                    square = True, cmap='viridis', ax = axes[1,1])

        axes[0,0].set_ylabel('Native space')
        axes[1,0].set_ylabel('Shared space')
        axes[1,0].set_xlabel('Within-subject correlation')
        axes[1,1].set_xlabel('Inter-subject correlation')
        f.suptitle('Representational similarity matrix, layer %d' % l, y=1.02)
        f.tight_layout()

        # save plot
        fig_name = 'wirsa_ns_l%.2d' % l
        fig_spath = os.path.join(plt_dir, fig_name)
        f.savefig(fig_spath, bbox_inches="tight")
        f.clf()
