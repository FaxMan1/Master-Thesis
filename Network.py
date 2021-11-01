#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


# # The Framework
class NeuralNetwork:
    
    def __init__(self, sizes, lr=0.01, type='classification', afunc='tanh', startweights=None, startbiases=None):
    
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(1,y) for y in sizes[1:]]                     # List of bias-lists.
        self.weights = [np.random.randn(x,y)/np.sqrt(x) for x,y in zip(sizes[:-1], sizes[1:])] # List of weight-matrices
        # self.biases = [np.random.uniform(bounds[0], bounds[1], (1,y)) for y in sizes[1:]]
        # self.weights = [np.random.uniform(bounds[0], bounds[1], (x, y)) for x,y in zip(sizes[:-1], sizes[1:])]
        self.lr = lr
        self.type = type
        if afunc == 'tanh':
            self.afunction = ActivationFunctions.tanh
            self.derivfunction = ActivationFunctions.tanhPrime
        elif afunc == 'relu':
            self.afunction = ActivationFunctions.ReLu
            self.afunction = ActivationFunctions.ReLuPrime
        
        # It is important that startweights and self.weights have the same dimensions!!
        if (startweights != None):
            self.weights = startweights
        
        # Same for bias
        if (startbiases != None):
            self.biases = startbiases
        
        pass
    
    
    def feedforward(self, a):
        
        self.activations = [a]            # List to hold activations
        self.zs = []                      # List to hold induced local fields
        self.a = np.array(a, ndmin = 2)
        
        # Feedforward
        for b,w in zip(self.biases, self.weights):
            self.z = np.dot(self.a, w) + b
            self.a = self.afunction(self.z)
            self.zs.append(self.z)
            self.activations.append(self.a)

        if self.type == 'classification':
            # softmax for classification
            return ActivationFunctions.stablesoftmax(self.z)

        elif self.type == 'regression':
            # Linear output layer
            return self.z

    
    def train(self, X, y):
        
        n = self.num_layers - 1         # n is the index of output activation
        X = np.array(X, ndmin = 2)      # Convert to 2D Numpy for good measure
        changesw = []
        changesb = []
        
        self.yhat = self.feedforward(X) # Forward Propagate
        
        
        ##### Backpropagation ####
        
        # Output Layer
        delta = CostFunctions.derivCE(self.yhat, y)
        dJdW = np.dot(self.activations[n-1].T, delta)
        dJdB = np.sum(delta, axis = 0)
        changesw.insert(0, dJdW)
        changesb.insert(0, dJdB)
        
    
        # Hidden Layers
        p = 2
        for w in reversed(self.weights[1:]):
            
            delta = np.dot(delta, w.T) * self.derivfunction(self.zs[n-p])
            dJdW = np.dot(self.activations[n-p].T, delta) 
            dJdB = np.sum(delta, axis = 0)
            changesw.insert(0, dJdW)
            changesb.insert(0, dJdB)
            p = p + 1
            
            
        
        ####  Gradient Descent ####
        
        # number of training examples
        m = y.shape[0]

        for i in range(n):
            self.weights[i] = self.weights[i] - (1.0/m) * self.lr * changesw[i]
            self.biases[i] = self.biases[i] - (1.0/m) * self.lr * changesb[i]
        
        
        pass
    
    
    def MSE(self, X, y):

        # Forward Propagate
        yhat = self.feedforward(X)

        # Calculate MSE
        return np.mean((y-yhat)**2) # * 0.5

    

    def train_loop(self, Xtrain, Ytrain, Xtest, Ytest, epochs, batch_size, with_tqdm=False, cost = False, cost_last = False, acc = False, acc_last = False):
        iterations_per_epoch = Xtrain.shape[0] // batch_size
        testcosts = np.zeros(iterations_per_epoch)
        testaccs = np.zeros(iterations_per_epoch)
        trainingcosts = np.zeros(iterations_per_epoch)
        epochcosts = np.zeros((epochs, len(testcosts)))
        epochaccs = np.zeros((epochs, len(testcosts)))

        idx = np.arange(Xtrain.shape[0])

        for k in range(epochs):
            if with_tqdm:
                #use tqdm
                for i in tqdm(range(iterations_per_epoch)):

                    chosen = np.random.choice(idx, batch_size, replace=False)
                    Xbatch = Xtrain[chosen]
                    Ybatch = Ytrain[chosen]

                    self.train(Xbatch, Ybatch)
                    if cost:
                        testcosts[i] = self.MSE(Xtest, Ytest)
                    elif acc:
                        testaccs[i] = accuracy(Xtest, Ytest)
            else:
                print("{}/{}".format(k+1, epochs), end="\r", flush=True)
                #no tqdm
                for i in range(iterations_per_epoch):
                    
                    if batch_size == Xtrain.shape[0]:
                        Xbatch = Xtrain
                        Ybatch = Ytrain
                    else:  
                        chosen = np.random.choice(idx, batch_size, replace = False)
                        Xbatch = Xtrain[chosen]
                        Ybatch = Ytrain[chosen]

                    self.train(Xbatch, Ybatch)
                    if cost:
                        testcosts[i] = self.MSE(Xtest, Ytest)
                    elif acc:
                        testaccs[i] = accuracy(Xtest, Ytest)

            
            epochcosts[k] = testcosts
            epochaccs[k] = testaccs
            #print(accuracy(XtestMnist, YtestMnist, "MNIST"))
            
        if cost_last:
            return CostFunctions.crossEntropy(Xtest, Ytest)
        
        if acc_last:
            return accuracy(Xtest, Ytest)
        
        if acc:
            return epochaccs.mean(axis=1)
        
        return epochcosts.mean(axis=1)
    


# # The Functions
class CostFunctions:
    
    @staticmethod
    def MSE(NN, X, y):
        
        # Forward Propagate
        yhat = NN.feedforward(X)
    
        # Calculate MSE
        return np.mean((y-yhat)**2) # * 0.5
    
    def RMSE(X, y):
        
        yhat = NN.feedforward(X)
        
        return np.square(np.sum(pow(yhat - y, 2)) / yhat.shape[0])
    
    @staticmethod
    def derivMSE(yhat, y):
        
        return (yhat - y)


    @staticmethod
    def crossEntropy(X, y, epsilon = 1e-15):
        
        #Forward Propagate
        a = NN.feedforward(X)
        
        # Calculate CrossEntropy error
        return np.mean(np.nan_to_num(-y*np.log(a+epsilon)-(1-y)*np.log((1-a)+epsilon)))
    
    
    @staticmethod
    def crossEntropy2(X, y):
        
        yhat = NN.feedforward(X)
        
        return -np.dot(np.divide(1, yhat.shape[0]), np.sum(np.multiply(y, np.log(yhat, out=np.zeros_like(yhat), where=y!=0))))
    
    #np.divide(cm, sum_rows, out=np.zeros_like(cm), where=sum_rows!=0)
    
    @staticmethod
    def derivCE(yhat, y):
        
        return (yhat - y)
    


class ActivationFunctions:
    
    @staticmethod
    def tanh(z):
        
        return np.tanh(z)
    
    @staticmethod
    def tanhPrime(z):
        
        return 1 - (np.tanh(z)**2)
        
    @staticmethod
    def ReLu(z, alpha = 0):
        
        return np.maximum(alpha*z, z)
        #return np.maximum(0, z)
    
    @staticmethod
    def ReLuPrime(z, alpha = 0):
        
        # return alpha if x < 0 else 1
        dx = np.ones_like(z)
        dx[z < 0] = alpha
            
        return dx
    
    
    @staticmethod
    def softmax(z):
        
        exps = np.exp(z).T
        return (exps / np.sum(exps, axis = 0)).T
    
    #def stablesoftmax(z):
        
        #exps = np.exp(z - np.max(z)).T
        #return (exps / np.sum(exps, axis = 0)).T
        
    @staticmethod
    def stablesoftmax(x):
    
        mx = np.max(x, axis=-1, keepdims=True)
        numerator = np.exp(x - mx).round(3)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        return numerator/denominator
    
    
    @staticmethod
    def sigmoid(z):
        
        return (1.0/(1.0+np.exp(-z)))
    
    @staticmethod
    def sigmoidPrime(z):
        
        return np.exp(-z)/((1+np.exp(-z))**2)


def accuracy(Xtest, Ytest, problem = None, plot = False):
    
    predictions = NN.feedforward(Xtest).argmax(axis = 1)
    true = Ytest.argmax(axis = 1)
    correct_preds = np.equal(true, predictions)
    
    if (plot == True):
    
        if (problem == "IRIS"):
            classes = np.array(['setosa', 'versicolor', 'virginica'])
        elif (problem == "MNIST"):
            classes = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        elif (problem == "XOR"):
            classes = np.array(["0", "1"])


        print("Accuracy: ", (sum(correct_preds) / len(true)) * 100, "%")
        print("Cross Entropy:", CostFunctions.crossEntropy(Xtest, Ytest))
        plotConfusionMatrix(true, predictions, classes, title = "Confusion Matrix", normalize = True, figsize = (7,7))
        plt.show()
    
    return (sum(correct_preds) / len(true))



def Kfold(X, splits=10):
    n = X.shape[0]  # n is the number of examples
    indices = set(np.arange(n)) # Creates a set of indices for the data

    test_n = [n // splits] * splits # num samples for each split from integer divison
    mod = n % splits    # the remainder from the integer division
    if mod > 0 and mod < splits:    # if remainder is positive, then distribute it as evenly as possible to each split
        for i in range(mod):
            test_n[i] += 1

    idx_ranges = []
    start = 0
    for end in test_n:
        idx_ranges.append((start, start + end))
        start = start + end

    all_train_indices = []
    all_test_indices = []
    for start, end in idx_ranges:
        test_indices = set(range(start, end))
        all_test_indices.append(list(test_indices))

        train_indices = indices.difference(test_indices)
        all_train_indices.append(list(train_indices))

    return zip(all_train_indices, all_test_indices)


### Function to plot Confusion Matrix ###
def plotConfusionMatrix(true, predictions, classes, title, normalize = True, figsize = (5,5)):

    cm = confusion_matrix(true, predictions) # Create the Confusion Matrix as numpy array
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalizes
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title)
           #ylabel='True label',
           #xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    
    ax.set_ylabel('True label', fontsize = 16.0)
    ax.set_xlabel('Predicted label', fontsize = 16.0)
    ax.set_title(title, fontsize = 15.0)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    return ax

np.set_printoptions(precision=2)

plt.show()
