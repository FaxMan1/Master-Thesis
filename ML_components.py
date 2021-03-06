import numpy as np
from Network import NeuralNetwork
import torch
import torchaudio
from filters import butter_lowpass

def load_params(weight_file, bias_file):
    w_container = np.load(weight_file)
    b_container = np.load(bias_file)
    weights = [w_container[key] for key in w_container]
    biases = [b_container[key] for key in b_container]
    sizes = [weights[0].shape[0]]

    for i in range(len(weights)):
        sizes.append(weights[i].shape[1])

    return weights, biases, sizes


def network_sender_receiver(upsampled, classes, sigma=0.89, cutoff_freq=2, path='../Joint_Models/', models=None, lowpass='butter'):

    classes = np.array(classes)
    upsampled = torch.Tensor(upsampled).view(1, 1, -1)

    if models is None:
        NN_tx = torch.load(path + 'DE_SenderButter0675_Best')
        NN_rx = torch.load(path + 'DE_ReceiverButter0675_Best')
    else:
        NN_tx, NN_rx = models

    if lowpass == 'butter':
        b, a = butter_lowpass(cutoff_freq, 8, 10) # 8 er sample rate. Skal ændres, hvis vi bruger en anden sample rate
        b = torch.tensor(b, requires_grad=True).float()
        a = torch.tensor(a, requires_grad=True).float()

    Tx = NN_tx(upsampled)

    if lowpass == 'butter':
        # Send filtered signal through lowpass filter
        Tx = torchaudio.functional.filtfilt(Tx, a, b)
    elif lowpass == 'ideal':
        Tx_freq = torch.fft.rfft(Tx)
        xf = torch.fft.rfftfreq(Tx.shape[2], 1 / 8)
        Tx_freq[0][0][xf > cutoff_freq] = 0
        Tx = torch.fft.irfft(Tx_freq, n=Tx.shape[2])

    # Normalize signal
    Tx = Tx / torch.sqrt(torch.mean(torch.square(Tx)))
    # Transmit signal
    Tx = Tx + torch.normal(0.0, sigma, Tx.shape)

    output = NN_rx(Tx)[0].T
    decisions = classes[output.argmax(axis=1)]

    return decisions

def ML_filtering(blocks, classes, model=None):

    X = np.array(blocks)
    classes = np.array(classes)
    if model is not None:
        return classes[model.feedforward(X).argmax(axis=1)]

    w, b, sizes = load_params('../Weights/best_filter_weights_8.npz',
                              '../Weights/best_filter_biases_8.npz')
    best_agent = NeuralNetwork(sizes, startweights=w, startbiases=b,
                               type='classification', afunc='relu')

    return classes[best_agent.feedforward(X).argmax(axis=1)]


def network_receiver(Rx, classes, model=None, path='../Conv1DModels/'):

    X = torch.tensor(Rx).view(1, 1, -1).float()
    classes = np.array(classes)

    if model is None:
        model = torch.load(path + 'Best_DE_Model2')

    return classes[model(X).argmax(axis=1)]


def ML_downsampling(blocks, classes, model=None):

    X = np.array(blocks)
    classes = np.array(classes)

    if model is not None:
        return classes[model.feedforward(X).argmax(axis=1)]

    w, b, sizes = load_params('../Weights/block_decision_making_weights2.npz',
                             '../Weights/block_decision_making_biases2.npz')
    best_agent = NeuralNetwork(sizes, startweights=w, startbiases=b,
                              type='classification', afunc='relu')

    return classes[best_agent.feedforward(X).argmax(axis=1)]


def ML_decision_making(downsampled, classes, model=None,
                       w_path='../Weights/decision_making_weights.npz',
                       b_path='../Weights/decision_making_biases.npz'):

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
