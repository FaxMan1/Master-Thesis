import torch
import torch.nn
import torchaudio
import matplotlib.pyplot as plt
from Comms_System import butter_lowpass
import numpy as np

def compute_loss(model, X, y, criterion):
    output = model(X)[0].T
    loss = criterion(output, y)
    return loss.item()

def train_loop(model, optimizer, cost, Xtrain, ytrain, Xtest=None, ytest=None, epochs=100,
                        use_cuda=False, eval=False, plot_iteration=1000):

    # device = torch.device('cuda') if torch.cuda_available else torch.device('cpu')
    if torch.cuda.is_available and use_cuda:
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    Xtrain, ytrain = Xtrain.to(device), ytrain.to(device)
    Xtest, ytest = Xtest.to(device), ytest.to(device)

    epoch_losses_train = torch.zeros(epochs)
    epoch_losses_test = torch.zeros(epochs)

    for i in range(epochs):
        model.train()
        inputs = Xtrain #.view(1, 1, -1).float()
        labels = ytrain

        optimizer.zero_grad()

        output = model(inputs)[0].T
        loss = cost(output, labels.long())
        loss.backward()
        optimizer.step()
        # model.

        if eval:
            model.eval()
            with torch.no_grad():

                test_loss = compute_loss(model, Xtest,
                                         ytest.long(), cost)

                epoch_losses_train[i] = loss.item()
                epoch_losses_test[i] = test_loss

                if i % plot_iteration == 0:

                    print("Epoch %d: \n Train Loss: %s \n Test Loss: %s" %
                        (i, loss.item(), test_loss))

                    plt.plot(list(model.parameters())[0].cpu().detach()[0][0])
                    # plt.plot(list(NN.parameters())[1].detach())
                    plt.show()
                    #print(loss.item())

    return epoch_losses_test, epoch_losses_train


def train_loop_minibatch(model, optimizer, cost, Xtrain, ytrain, Xtest=None, ytest=None, batch_size=200, epochs=100,
                        use_cuda=False, eval=False, plot_iteration=1000):

    # device = torch.device('cuda') if torch.cuda_available else torch.device('cpu')
    if torch.cuda.is_available and use_cuda:
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    Xtrain, ytrain = Xtrain.to(device), ytrain.to(device)
    Xtest, ytest = Xtest.to(device), ytest.to(device)
    model = model.to(device)

    iterations_per_epoch = Xtrain.shape[0] // batch_size
    idx = torch.arange(Xtrain.shape[0])

    epoch_losses_train = torch.zeros(epochs)
    epoch_losses_test = torch.zeros(epochs)

    for i in range(epochs):
        model.train()
        for j in range(iterations_per_epoch):
            print(j)
            if batch_size == Xtrain.shape[0]:
                Xbatch = Xtrain.view(1, 1, -1).float()
                ybatch = ytrain
            else:
                #  chosen = np.random.choice(idx, batch_size, replace=False)
                chosen = torch.multinomial(idx.float(), batch_size, replacement=False)
                Xbatch = Xtrain[chosen].view(1, 1, -1).float()  # reshape and cast to float so PyTorch understands it
                ybatch = ytrain[chosen//8]

            optimizer.zero_grad()

            output = model(Xbatch)[0].T
            loss = cost(output, ybatch.long())
            loss.backward()
            optimizer.step()

        if eval:
            model.eval()
            with torch.no_grad():

                train_loss = compute_loss(model, Xtrain.view(1, 1, -1).float(),
                                          ytrain.long(), cost)
                test_loss = compute_loss(model, Xtest.view(1, 1, -1).float(),
                                         ytest.long(), cost)

                epoch_losses_train[i] = train_loss
                epoch_losses_test[i] = test_loss

                if i % 1000 == 0:

                    print("Epoch %d: \n Train Loss: %s \n Test Loss: %s" %
                        (i, train_loss, test_loss))

                if i % plot_iteration == 0:
                    plt.plot(list(model.parameters())[0].to('cpu').detach()[0][0])
                    # plt.plot(list(NN.parameters())[1].detach())
                    plt.show()
                    print(loss.item())

    return epoch_losses_test, epoch_losses_train


def joint_train_loop(NN_tx, NN_rx, X, y, optimizer, cost, epochs=100, SNRdb=10, lowpass=False, cutoff_freq=1,
                     plot_iteration=300, sample_rate=8, use_cuda=False, v=False):

    SNR = 10 ** (SNRdb / 10)
    sigma = np.sqrt(8 / SNR)
    print(sigma)

    if torch.cuda.is_available and use_cuda:
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    NN_tx, NN_rx = NN_tx.to(device), NN_rx.to(device)
    X, y = X.to(device), y.long().to(device)

    epoch_losses_train = torch.zeros(epochs).to(device)
    b, a = butter_lowpass(cutoff_freq, sample_rate, 4)
    b = torch.tensor(b, requires_grad=True).float().to(device)
    a = torch.tensor(a, requires_grad=True).float().to(device)

    for i in range(epochs):

        Tx = NN_tx(X)
        # Send filtered signal through lowpass filter
        Tx_low = torchaudio.functional.lfilter(Tx, a, b)
        Tx_low = Tx_low / torch.sqrt(torch.mean(torch.square(Tx_low))) # normalize
        Tx_low = Tx_low + torch.normal(0.0, sigma, Tx_low.shape, device=device)
        received = NN_rx(Tx_low)[0].T

        optimizer.zero_grad()
        loss = cost(received, y)
        loss.backward()
        optimizer.step()

        epoch_losses_train[i] = loss.item()

        if v and (i + 1) % plot_iteration == 0:
            print(i, " loss:", loss.item())
            acc = torch.sum(received.argmax(axis=1) == y) / len(y)
            print(i, " acc:", acc.item())

            plt.figure()
            plt.title('Sender Weights')
            plt.plot(list(NN_tx.parameters())[0].to('cpu').detach()[0][0])
            plt.show()
            plt.magnitude_spectrum(list(NN_tx.parameters())[0].to('cpu').detach()[0][0], Fs=sample_rate, color='C1')
            plt.show()

            plt.figure()
            plt.title('Receiver Weights')
            plt.plot(list(NN_rx.parameters())[0].to('cpu').detach()[0][0])
            plt.show()
            plt.magnitude_spectrum(list(NN_rx.parameters())[0].to('cpu').detach()[0][0], Fs=sample_rate, color='C1')
            plt.show()


    return epoch_losses_train


def train_loop_old(model, optimizer, criterion, X, y, epochs=1000, v=False, plot_iteration=1000):
    X = X.view(1, 1, -1).float()

    for i in range(epochs):
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)[0].T
        labels = y.long()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if v:
            if i % plot_iteration == 0:
                plt.plot(list(model.parameters())[0].detach()[0][0])
                # plt.plot(list(NN.parameters())[1].detach())
                plt.show()
                print(loss.item())