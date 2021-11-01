#!/usr/bin/env python
# coding: utf-8

from DE_minibatch import DE
from objective_functions import MSE
from data import create_3d_gauss_data
from visualizer import plot_2d_gauss
from Comms_System import Comms_System
import numpy as np
# %%
Xtrain, ytrain, Xtest, ytest = create_3d_gauss_data()

D = DE(objective_function=MSE, sizes=[2,20,1], pop_size=50, F=0.55, cr=0.85, X=Xtrain, y=ytrain)
D.evolution(num_epochs=2001, batch_size=100, verbose=True)
D.evaluate(xtest=Xtest, plot_function=plot_2d_gauss)

# %%


