import numpy as np
from Network import NeuralNetwork
# from train_decision_making import generate_train_data, train

def load_params(weight_file, bias_file):
    w_container = np.load(weight_file)
    b_container = np.load(bias_file)
    weights = [w_container[key] for key in w_container]
    biases = [b_container[key] for key in b_container]
    sizes = [weights[0].shape[0]]

    for i in range(len(weights)):
        sizes.append(weights[i].shape[1])

    return weights, biases, sizes


def ML_decision_making(downsampled, classes, mode='load', type='CE'):

    weights, biases, sizes = load_params('../SavedWeights/decision_making_weights_CE.npz',
                                         '../SavedWeights/decision_making_biases_CE.npz')
    weights2, biases2, sizes2 = load_params('../SavedWeights/decision_making_weights_MSE.npz',
                                         '../SavedWeights/decision_making_biases_MSE.npz')
    best_agent_CE = NeuralNetwork(sizes, startweights=weights, startbiases=biases,
                               type='classification', afunc='relu')
    best_agent_MSE = NeuralNetwork(sizes2, startweights=weights2, startbiases=biases2,
                                  type='classification', afunc='relu')

    X = np.array(downsampled, ndmin=2).T
    classes = np.array(classes)
    if type == 'CE':
        NN_decisions = classes[best_agent_CE.feedforward(X).argmax(axis=1)]
    elif type == 'MSE':
        NN_decisions = classes[best_agent_MSE.feedforward(X).argmax(axis=1)]

    return NN_decisions
