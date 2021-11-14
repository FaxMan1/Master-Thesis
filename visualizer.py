import numpy as np
import matplotlib.pyplot as plt
from objective_functions import gaussian_2d_old, gauss_2d


def plot_simple_data(agent, Xtest, ytest, title='Approximated Function', savefig=False):
    plt.figure(figsize=(13, 8))
    yhat = agent.feedforward(Xtest)
    # plt.plot(xtrain, ytrain, 'kx', label = 'Training Data')
    plt.plot(Xtest, ytest, 'bx', label='Test Data')
    plt.plot(Xtest, yhat, '-r', label='Learned Function')
    # plt.title(r'Fit of $y = x^2$', fontsize = 15)
    plt.xlabel('x', fontsize=24)
    plt.ylabel('y', fontsize=24)
    plt.legend(fontsize=14)
    if savefig:
        plt.savefig(title)
    else:
        plt.show()


def prep_for_2d_mesh(agent, Xtest):
    x1, x2 = Xtest[:, 0], Xtest[:, 1]
    x1mesh, x2mesh = np.meshgrid(x1, x2)
    yhats = []
    xtest3d = []

    for x1_, x2_ in zip(x1mesh, x2mesh):
        xtest3d.append(np.column_stack([x1_, x2_]))

    for x in xtest3d:
        yhat = agent.feedforward(x)
        yhats.append(yhat.flatten())

    yhats = np.array(yhats)
    return yhats


def plot_ackley(func, agent, bounds=None, title=''):
    if bounds is not None:
        lb, ub = bounds
    else:
        ub = abs(1.5 * agent[0])
        lb = -ub

    px, py = agent

    x_span = np.linspace(lb, ub, 50)
    y_span = np.linspace(lb, ub, 50)
    x, y = np.meshgrid(x_span, y_span)
    z = func([x, y])  # formula for parabola in 3D
    pz = func([px, py])

    plt.figure(figsize=(16, 10))
    ax = plt.axes(projection='3d')
    ax.scatter(px, py, pz, s=100, color='red');
    ax.plot_surface(x, y, z, cmap='viridis', alpha=0.75)
    # ax.plot_wireframe(x, y, z, cmap='viridis',zorder=1)
    ax.set_title(title, fontsize=26);
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('z', fontsize=20)
    plt.show()


def plot_2d_gauss(agent, X=None, ytest=None, bounds=[-1, 1], mu=[0, 0], sigma=[0.25, 0.25], verbose=False, title=' ',
                  savefig=False):

    y = prep_for_2d_mesh(agent, X)

    if X is None:
        x1 = np.linspace(bounds[0], bounds[1], 50)
        x2 = np.linspace(bounds[0], bounds[1], 50)
        x1, x2 = np.meshgrid(x1, x2)
        y = gaussian_2d_old([x1, x2], mu, sigma)
    elif X is not None:
        x1, x2 = X[:, 0], X[:, 1]
        x1, x2 = np.meshgrid(x1, x2)
        if y is None:
            y = gaussian_2d_old([x1, x2], mu, sigma)

    plt.figure(figsize=(16, 10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(x1, x2, y, cmap='viridis', alpha=1)
    ax.set_title(title, fontsize=26);
    ax.set_xlabel('x1', fontsize=20)
    ax.set_ylabel('x2', fontsize=20)
    ax.set_zlabel('y', fontsize=20)
    if savefig:
        plt.savefig(title)
    else:
        plt.show()