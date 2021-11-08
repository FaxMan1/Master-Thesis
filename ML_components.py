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


def ML_downsampling(blocks, classes, model=None):

   X = np.array(blocks)
   classes = np.array(classes)
   w, b, sizes = load_params('../SavedWeights/block_decision_making_weights.npz',
                             '../SavedWeights/block_decision_making_biases.npz')
   best_agent = NeuralNetwork(sizes, startweights=w, startbiases=b,
                              type='classification', afunc='relu')

   return classes[best_agent.feedforward(X).argmax(axis=1)]




def ML_decision_making(downsampled, classes, model=None,
                       w_path='../SavedWeights/decision_making_weights.npz',
                       b_path='../SavedWeights/decision_making_biases.npz'):

    X = np.array(downsampled, ndmin=2).T
    classes = np.array(classes)

    if model is not None:
        NN_decisions = classes[model.feedforward(X).argmax(axis=1)]
    else:
        weights, biases, sizes = load_params(w_path, b_path)
        best_agent = NeuralNetwork(sizes, startweights=weights, startbiases=biases,
                                   type='classification', afunc='relu')
        NN_decisions = classes[best_agent.feedforward(X).argmax(axis=1)]

    return NN_decisions
