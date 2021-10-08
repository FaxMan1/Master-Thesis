import numpy as np
from scipy.stats import multivariate_normal


def MSE(theta):
    y, yhat = theta

    return np.mean((y - yhat) ** 2)


def paraboloid(theta, param=[0, 0]):
    x, y = theta
    h, k = param

    return (x - h) ** 2 + (y - k) ** 2


def ackley(theta, param=None):
    x, y = theta
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20


# define normalized 2D gaussian
def gaussian_2d_old(theta, mu=[0, 0], sigma=[0.25, 0.25]):
    x, y = theta
    mx, my = mu
    sx, sy = sigma
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))


def gauss_2d(X, mu=[0, 0], sigma=[0.25, 0.25]):
    var = multivariate_normal(mean=mu, cov=np.diag(sigma))
    return var.pdf(X)