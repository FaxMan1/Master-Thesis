from Comms_System import Comms_System
import numpy as np
from DE_minibatch import DE
from objective_functions import MSE
from objective_functions import crossEntropy

# %%

def generate_train_data(num_symbols):

    symbol_set = [3, 1, -1, -3]  # all symbols that we use
    symbol_seq = np.random.choice(symbol_set, num_symbols, replace=True)
    m = 8
    CS = Comms_System(symbol_set=symbol_set, symbol_seq=symbol_seq, num_samples=m)

    # Automatic test
    _, _,_, downsampled = CS.test_CS(noise_level=1, v=False)

    return np.array(downsampled, ndmin=2).T, symbol_seq, np.array(symbol_set)

# %%

def train_DM_model(sizes=[1,8,4], epochs=501, num_symbols=1000, splitlen=0.75, verbose=True, saveweights=False):

    X, y, classes = generate_train_data(num_symbols)
    num_classes = len(classes)
    sizes[-1] = num_classes

    # Makes a dictionary that maps a number to each symbol e.g. 3: 0
    class_idx = {v: i for i, v in enumerate(classes)}

    # Maps class indexes to each value in targets
    y = np.array([class_idx[v] for v in y])

    # Converts to one-hot-encoded
    y = np.eye(num_classes)[y]

    # split train-test
    splitlen = int(X.shape[0] * 0.75)
    Xtrain, ytrain = X[:splitlen], y[:splitlen]
    Xtest, ytest = X[splitlen:], y[splitlen:]

    D = DE(objective_function=crossEntropy, sizes=sizes, pop_size=50, F=0.55, cr=0.85,
           X=Xtrain, y=ytrain, type='classification', afunc='relu')

    best_agent = D.evolution(num_epochs=epochs, batch_size=Xtrain.shape[0], verbose=verbose, print_epoch=100)
    D.evaluate(xtest=Xtest)

    if verbose:
        # Get predictions
        predictions = best_agent.feedforward(Xtest).argmax(axis=1)
        true = ytest.argmax(axis=1)

        # Convert back to symbols
        predicted_symbols = classes[predictions]
        true_values = classes[true]
        # print(predicted_symbols, true_values)
        correct_preds = np.equal(true_values, predicted_symbols)
        print("Accuracy: ", (sum(correct_preds) / len(true_values)) * 100, "%")

    if saveweights:
        D.save_params('decision_making_weights', 'decision_making_biases')

    return D


