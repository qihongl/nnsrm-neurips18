import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict


class Net(torch.nn.Module):
    """a neural network with 1 sigmoid hidden layer, and a linear output layer
    """

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, 1)   # output layer

    def forward(self, x):
        h = torch.sigmoid(self.hidden(x))      # activation function for hidden layer
        output = self.out(h)
        return output, h


def train_net(net, loss_func, optimizer, n_epochs,
              x_train, y_train, x_test, y_test):
    # input:
    # - net: a pytorch network
    # - x_train: training set data
    # - y_train: training set labels
    # - x_test: test set data
    # - y_test: test set labels
    # action:
    # - train the network
    # return:
    # - net: a trained network
    # - the accuracy scores for the training/test set
    #
    n_examples = len(y_train)
    accuracy_train = np.zeros(n_epochs, )
    accuracy_test = np.zeros(n_epochs, )
    for t in range(n_epochs):
        out, h = net(x_train)
        out = torch.squeeze(out)
        loss = loss_func(out, y_train)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        # record learning process
        y_hat = torch.round(out).data.numpy()
        y_target = y_train.data.numpy()
        accuracy_train[t] = sum(y_hat == y_target) / n_examples
        accuracy_test[t] = test_net(net, x_test, y_test)
    return net, accuracy_train, accuracy_test


def test_net(net, x, y):
    # given a network and some training data
    # evaluate the performance in terms of classification accuracy
    #
    n_examples = len(y)
    # compute the prediction
    out, _ = net(x)
    out = torch.squeeze(out)
    y_hat = torch.round(out).data.numpy()
    # get the targets
    y_target = y.data.numpy()
    # compute the accuracy
    accuracy = sum(y_hat == y_target) / n_examples
    return accuracy


def get_hidden_acts(input_net, x):
    # given a network and some data
    # fetch the hidden activity matrix
    _, h = input_net(x)
    return h.data.numpy()


def generate_permuted_params(net, n_hidden):
    # input: a 2 layered standard network inpytorch
    # output: its permutated weights

    # read the weights
    h_wts, h_bias, o_wts, o_bias = get_wts(net)
    # generate a random permutation
    perm = np.random.permutation(n_hidden)
    # pack the weights into a ordered dict
    param_dict = pack_params(
        h_wts[perm, :], h_bias[[perm]], o_wts[:, perm], o_bias)
    return param_dict, perm


def get_wts(net):
    # assuming pytorch net with 1 hidden layer
    # read the weights
    h_wts = net.state_dict()['hidden.weight']
    h_bias = net.state_dict()['hidden.bias']
    o_wts = net.state_dict()['out.weight']
    o_bias = net.state_dict()['out.bias']
    return h_wts, h_bias, o_wts, o_bias


def pack_params(h_wts_perm, h_bias_perm,
                o_wts_perm, o_bias_perm):
    # pack network parameters to a dict
    param_dict = OrderedDict()
    param_dict = {'hidden.weight': h_wts_perm,
                  'hidden.bias': h_bias_perm,
                  'out.weight': o_wts_perm,
                  'out.bias': o_bias_perm}
    return param_dict
