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


def ML_decision_making(downsampled, classes, mode='load'):


    if mode == 'load':
        weights, biases, sizes = load_params('../SavedWeights/decision_making_weights_1.11_10.9.npz',
                                             '../SavedWeights/decision_making_biases_1.11_10.9.npz')
        best_agent = NeuralNetwork(sizes, startweights=weights, startbiases=biases,
                                   type='classification', afunc='relu')

    # elif mode == 'train':
        # downsampled, y, classes = generate_train_data()
        # best_agent = train(downsampled, y, classes)

    X = np.array(downsampled, ndmin=2).T
    classes = np.array(classes)
    NN_decisions = classes[best_agent.feedforward(X).argmax(axis=1)]

    return NN_decisions
