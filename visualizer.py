import numpy as np
import matplotlib.pyplot as plt
from objective_functions import gaussian_2d_old, gauss_2d


def plot_2d_gauss(X=None, y=None, bounds=[-1, 1], mu=[0, 0], sigma=[0.25, 0.25], verbose=False, title=' ',
                  savefig=False):
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