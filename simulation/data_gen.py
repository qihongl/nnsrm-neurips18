import torch
from torch.autograd import Variable
import numpy as np
# torch.manual_seed(0)    # reproducible


def get_XOR_data(n_data=100):
    n_features = 2
    mean = torch.ones(n_data, n_features)
    variance = .3

    # specify the center
    x0 = torch.cat((torch.ones(n_data, 1), -torch.ones(n_data, 1)), 1)
    x1 = torch.cat((-torch.ones(n_data, 1), torch.ones(n_data, 1)), 1)

    x2 = torch.cat((torch.ones(n_data, 1), torch.ones(n_data, 1)), 1)
    x3 = torch.cat((-torch.ones(n_data, 1), -torch.ones(n_data, 1)), 1)

    # add noise
    x0 = x0 + torch.normal(mean, variance)
    x1 = x1 + torch.normal(mean, variance)
    x2 = x2 + torch.normal(mean, variance)
    x3 = x3 + torch.normal(mean, variance)

    y0 = torch.zeros(n_data)
    y1 = torch.zeros(n_data)
    y2 = torch.ones(n_data)
    y3 = torch.ones(n_data)

    x = torch.cat((x0, x1, x2, x3), 0).type(torch.FloatTensor)
    # float tensor for MSE loss
    y = torch.cat((y0, y1, y2, y3), ).type(torch.FloatTensor)
    # long tensorflow for cross ent loss
    # y = torch.cat((y0, y1, y2, y3), ).type(torch.LongTensor)

    x, y = Variable(x), Variable(y)
    return x, y


def get_toy_data(n_data=100):
    n_features = 2
    mean = torch.zeros(n_data, n_features)
    variance = .4

    # specify the center
    x0 = torch.cat((-torch.ones(n_data, 1), torch.ones(n_data, 1)), 1)
    x1 = torch.cat((torch.ones(n_data, 1), -torch.ones(n_data, 1)), 1)

    x2 = torch.cat((torch.zeros(n_data, 1), torch.zeros(n_data, 1)), 1)
    x3 = torch.cat((torch.zeros(n_data, 1), torch.zeros(n_data, 1)), 1)

    # add noise
    x0 = x0 + torch.normal(mean, variance)
    x1 = x1 + torch.normal(mean, variance)
    x2 = x2 + torch.normal(mean, variance)
    x3 = x3 + torch.normal(mean, variance)

    y0 = torch.zeros(n_data)
    y1 = torch.zeros(n_data)
    y2 = torch.ones(n_data)
    y3 = torch.ones(n_data)

    x = torch.cat((x0, x1, x2, x3), 0).type(torch.FloatTensor)
    # float tensor for MSE loss
    y = torch.cat((y0, y1, y2, y3), ).type(torch.FloatTensor)
    # long tensorflow for cross ent loss
    # y = torch.cat((y0, y1, y2, y3), ).type(torch.LongTensor)

    x, y = Variable(x), Variable(y)
    return x, y


def get_wave_data(step_size):
    noise_mean = 0
    noise_variance = .1
    x_range = np.arange(start=0, stop=2 * np.pi, step=step_size)
    # generate x and y
    x_true = np.concatenate([x_range, x_range + 2 * np.pi])
    y_true = np.concatenate([np.sin(x_range), np.cos(x_range)])
    # add noise
    noise = np.random.normal(loc=noise_mean, scale=noise_variance,
                             size=np.shape(y_true))
    y_obs = y_true + noise
    return x_true, y_obs, y_true


def get_wave_data(step_size):
    noise_mean = 0
    noise_variance = .1
    x_range = np.arange(start=0, stop=2 * np.pi, step=step_size)
    # generate x and y
    x_true = np.concatenate([x_range, x_range + 2 * np.pi])
    y_true = np.concatenate([np.sin(x_range), np.cos(x_range)])
    # add noise
    noise = np.random.normal(loc=noise_mean, scale=noise_variance,
                             size=np.shape(y_true))
    y_obs = y_true + noise
    return x_true, y_obs, y_true


def get_poly_data(step_size):
    noise_mean = 0
    noise_variance = .2
    x_range = np.arange(start=-.3, stop=2.5, step=step_size)
    # generate x and y
    x_true = x_range
    y_true = 2*x_true - 3 * x_true**2 + x_true**3
    # add noise
    noise = np.random.normal(loc=noise_mean, scale=noise_variance,
                             size=np.shape(y_true))
    y_obs = y_true + noise
    return x_true, y_obs, y_true
