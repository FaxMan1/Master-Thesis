import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from datetime import datetime
import copy
import torch
from filters import butter_lowpass, ideal_lowpass
import torchaudio
import random


class DE:

    def __init__(self, objective_function, pop_fun, X, y, pop_size=50,F=0.5, cr=0.5, use_cuda=False,
                 lowpass='butter', cutoff_freq=0.675, sample_rate=8, SNRdb=10, start_agent=None):
        if use_cuda and torch.cuda.is_available:
            self.device = torch.device('cuda')
            print('Using GPU')
        else:
            self.device = torch.device('cpu')
        self.obj = objective_function
        self.X = X.to(self.device)
        self.y = y.long().to(self.device)
        self.N = pop_size
        self.F = torch.Tensor([F]).to(self.device)
        self.cr = torch.Tensor([cr]).to(self.device)
        self.pop = [pop_fun() for i in range(pop_size)]
        self.pop = [(net_tx.to(self.device), net_rx.to(self.device)) for net_tx, net_rx in self.pop]
        self.testNN = (pop_fun()[0].to(self.device), pop_fun()[1].to(self.device))
        self.lowpass = lowpass
        self.cutoff = cutoff_freq
        self.sample_rate = sample_rate
        SNR = 10 ** (SNRdb / 10)
        self.sigma = np.sqrt(sample_rate / SNR)
        if start_agent is not None:
            self.pop[0] = copy.deepcopy(start_agent)

        if self.lowpass == 'butter':
            self.b, self.a = butter_lowpass(cutoff_freq, sample_rate, 10)
            self.b = torch.tensor(self.b).float().to(self.device)
            self.a = torch.tensor(self.a).float().to(self.device)

    def save_model(self, fname, path='../Joint_Models/', agent=None):
        if agent is None:
            agent = self.best_agent
        torch.save(agent, path+fname)

    def load_model(self, fname, path='../Joint_Models/'):
        return torch.load(path+fname)


    def get_x_chosen(self, y_chosen):
        y_chosen = y_chosen[y_chosen < self.y.shape[0]]
        y_start, y_end = y_chosen[0], y_chosen[-1]
        x_chosen = torch.arange(y_start * 8, (y_end + 1) * 8, device=self.device)
        x_chosen = x_chosen[x_chosen < self.X.shape[2]]
        return x_chosen


    def NN_obj(self, agent, x_chosen, y_chosen, noise=None):
        NN_tx, NN_rx = agent
        Tx = NN_tx(self.X[:,:,x_chosen])

        if self.lowpass == 'butter':
            Tx_low = torchaudio.functional.filtfilt(Tx, self.a, self.b)
        elif self.lowpass == 'ideal':
            Tx_freq = torch.fft.rfft(Tx)
            xf = torch.fft.rfftfreq(Tx.shape[2], 1 / self.sample_rate)
            Tx_freq[0][0][xf > self.cutoff] = 0
            Tx_low = torch.fft.irfft(Tx_freq, n=Tx.shape[2])

        Tx_low = Tx_low / torch.sqrt(torch.mean(torch.square(Tx_low)))  # normalize
        if noise is not None:
            Tx_low = Tx_low + self.noise_signal
        else:
            Tx_low = Tx_low + torch.normal(0.0, self.sigma, Tx_low.shape, device=self.device)
        received = NN_rx(Tx_low)[0].T

        return self.obj(received, self.y[y_chosen])

    def mutation(self, nets):

        for testp, p1, p2, p3 in zip(*[net.parameters() for net in nets]):
            testp.data = p1 + self.F * (p2 - p3)

        pass

    def crossover(self, donor, target):

        for dw, tw in zip(donor.parameters(), target.parameters()):
            crit = torch.rand(dw.shape, device=self.device) < self.cr
            trial_w = crit * dw + ~crit * tw
            dw.data = trial_w

        pass

    def evolve(self, x_chosen, y_chosen):

        for j, x in enumerate(self.pop):

            choice = np.random.choice(np.delete(np.arange(self.N), j), 3, replace=False)
            a, b, c = itemgetter(*choice)(self.pop)

            # Mutation
            self.mutation([self.testNN[0], a[0], b[0], c[0]])  # for Tx
            self.mutation([self.testNN[1], a[1], b[1], c[1]])  # for Rx

            # Crossover
            self.crossover(self.testNN[0], x[0])  # Tx
            self.crossover(self.testNN[1], x[1])  # Rx

            # Selection
            obj_u = self.NN_obj(self.testNN, x_chosen, y_chosen, noise=self.noise_signal)
            if obj_u < self.NN_obj(x, x_chosen, y_chosen, noise=self.noise_signal):
                self.pop[j] = copy.deepcopy(self.testNN)
                self.obj_all[j] = obj_u

    def evolution(self, num_epochs, batch_size, verbose=False, print_epoch=1000, print_mini=False, k_print=10):
        idx_x = np.arange(self.X.shape[2])
        idx_y = np.arange(self.y.shape[0])
        iterations_per_epoch = self.y.shape[0] // batch_size
        self.noise_signal = torch.normal(0.0, self.sigma, torch.Tensor(batch_size*8+63).shape, device=self.device)

        y_chosens = []
        for k in range(iterations_per_epoch):
            y_chosens.append(torch.arange(k * batch_size, (k + 1) * batch_size, device=self.device))

        # evaluate the initialized population with the objective function. Only RX agent is evaluated
        self.obj_all = torch.Tensor([self.NN_obj(agent, idx_x, idx_y) for agent in self.pop])

        # find the best agent within the initial population
        self.best_agent = self.pop[torch.argmin(self.obj_all)]

        best_obj = torch.min(self.obj_all)
        prev_obj = best_obj

        self.best_objs = np.zeros(num_epochs + 1)
        self.best_objs[0] = best_obj

        for i in range(num_epochs):
            #idx_y = torch.arange(self.y.shape[0], device=self.device).float()
            np.random.shuffle(y_chosens)
            for k in range(iterations_per_epoch):
                x_chosen = self.get_x_chosen(y_chosens[k])
                self.evolve(x_chosen, y_chosens[k])

                # update the current best objective function value
                best_obj = torch.min(self.obj_all)
                self.best_objs[i + 1] = best_obj
                if print_mini and k % k_print == 0:
                    print('%d: cost= %.5f' % (i, best_obj))

                if best_obj < prev_obj:
                    # update best agent
                    self.best_agent = self.pop[torch.argmin(self.obj_all)]
                    # update previous solution to use for next iteration
                    prev_obj = best_obj


            if verbose and i % print_epoch == 0:
                # report progress at each iteration
                print('%d: cost= %.5f' % (i, best_obj))
                #print('%d: acc= %.5f' % (i, self.accuracy(self.best_agent[1](self.X), self.y)))
                #plt.plot(list(self.best_agent[0].parameters())[0].cpu().detach()[0][0]) # Tx network parameters
                #plt.plot(list(self.best_agent[1].parameters())[0].cpu().detach()[0][0]) # Rx network parameters
                plt.show()

        return self.best_agent

    def evaluate(self, plot_function=None, agent=None, bounds=None, title=' '):

        if agent is None:
            agent = self.best_agent
            plt.figure(figsize=(13, 8))
            plt.plot(range(len(self.best_objs)), self.best_objs)
            plt.title('Training Graph', fontsize=24)
            plt.xlabel('Iterations', fontsize=20)
            plt.ylabel('Cost', fontsize=20)
            plt.legend(['Train', 'Test'], fontsize=14)
            plt.show()

        if plot_function is not None:
            plot_function(agent, self.Xtest, self.ytest, title=title, savefig=False)

        print(f"Best agent is {agent} with a train cost of {np.round(self.NN_obj(agent).cpu().detach(), 5)}.")

        # print(f"Worst initialization was {self.initial_worst_agent} with a cost of \
        # {np.round(self.obj(self.initial_worst_agent), 2)}.")

        pass

    def accuracy(self, predictions, ytest):
        predictions = predictions.argmax(axis=1)
        correct_preds = ytest == predictions
        return torch.sum(correct_preds) / len(ytest)

    def early_stop_training(self, patience, measure='cost', eval=True, v=True):

        n = 1
        iterations = 0
        if measure == 'cost':
            no_iterations_rising = 0
            val_error = 20000
            obj_all = torch.Tensor([self.NN_obj(agent) for agent in self.pop])
            self.opt_agent = copy.deepcopy(self.pop[torch.argmin(obj_all)])
            opt_iterations = iterations
            testcosts = []

            while (no_iterations_rising < patience):
                self.evolution(num_epochs=n, verbose=False, print_epoch=1)
                iterations = iterations + n
                val_error_new = self.obj(self.best_agent(self.Xtest)[0].T, self.ytest)
                testcosts.append(val_error_new.item())
                if (val_error_new < val_error):
                    if v: print(f"{iterations}: Test Cost Falling  {val_error_new}")
                    no_iterations_rising = 0
                    self.opt_agent = copy.deepcopy(self.best_agent)
                    opt_iterations = iterations
                    val_error = val_error_new
                else:
                    no_iterations_rising += n

            testcosts = np.array(testcosts)
            if v:
                print("Optimal number of iterations:", opt_iterations)
                print("Best error:", val_error)
                print("Error at stop:", val_error_new)

        elif measure == 'accuracy':
            no_iterations_falling = 0
            val_acc = 0
            opt_iterations = iterations
            testcosts = []
            obj_all = torch.Tensor([self.NN_obj(agent) for agent in self.pop])
            self.opt_agent = copy.deepcopy(self.pop[torch.argmin(obj_all)])

            while (no_iterations_falling < patience):
                self.evolution(num_epochs=n, verbose=False, print_epoch=1)
                iterations = iterations + n
                val_acc_new = self.accuracy(self.best_agent(self.Xtest), self.ytest)
                testcosts.append(val_acc_new.item())
                if (val_acc_new > val_acc):
                    if v: print(f"{iterations}: Test Accuracy Rising  {val_acc_new}")
                    no_iterations_falling = 0
                    self.opt_agent = copy.deepcopy(self.best_agent)
                    opt_iterations = iterations
                    val_acc = val_acc_new
                else:
                    no_iterations_falling += n
                    # print("Falling or the same")

            if v:
                print("Optimal number of iterations:", opt_iterations)
                print("Best accuracy:", val_acc)
                print("Accuracy at stop:", val_acc_new)

        if eval:
            plt.figure(figsize=(13, 8))
            plt.plot(testcosts)
            plt.title('Test Cost Graph', fontsize=24)
            plt.xlabel('Iterations', fontsize=20)
            plt.ylabel('Test Cost', fontsize=20)
            plt.show()

        return self.best_agent, self.opt_agent






