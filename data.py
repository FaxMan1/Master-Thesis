import numpy as np
from objective_functions import gauss_2d


def create_3d_gauss_data(mu=[0,0], sigma=[1,1], num_data_points=300):

    # create 2d/3d (???) gauss training data
    xtrain = np.random.multivariate_normal(mean=mu, cov=np.diag(sigma), size=num_data_points)
    ytrain = gauss_2d(xtrain, mu, sigma)
    ytrain = np.array(ytrain, ndmin=2).T

    # Create test data
    x_span = np.linspace(-1, 1, num_data_points)
    y_span = np.linspace(-1, 1, num_data_points)
    xtest = np.column_stack([x_span, y_span])
    ytest = gauss_2d(xtest, mu, sigma)
    ytest = np.array(ytest, ndmin=2).T

    return xtrain, ytrain, xtest, ytest


def create_simple_data(problem='parabola', num_data_points=30, bounds=None, noise=False):

    if problem == 'sine':
        Xtrain = np.random.uniform(low=-np.pi, high=np.pi, size=(num_data_points, 1))
        ytrain = np.sin(Xtrain)

        Xtest = np.linspace(-np.pi, np.pi, num_data_points)
        Xtest = np.array(Xtest, ndmin=2).T
        ytest = np.sin(Xtest)
        if noise:
            ytrain = ytrain + np.random.normal(0.0, 0.05, ytrain.shape)  # add gaussian noise
            ytest = ytest + np.random.normal(0.0, 0.05, ytest.shape) # add gaussian noise

        return Xtrain, ytrain, Xtest, ytest

    elif problem == 'parabola':
        Xtrain = np.random.uniform(low=-1, high=1, size=(num_data_points, 1))
        ytrain = Xtrain ** 2

        Xtest = np.linspace(-1, 1, num_data_points)
        Xtest = np.array(Xtest, ndmin=2).T
        ytest = Xtest ** 2
        if noise:
            ytrain = ytrain + np.random.normal(0.0, 0.05, ytrain.shape)  # add gaussian noise
            ytest = ytest + np.random.normal(0.0, 0.05, ytest.shape)  # add gaussian noise

    return Xtrain, ytrain, Xtest, ytest