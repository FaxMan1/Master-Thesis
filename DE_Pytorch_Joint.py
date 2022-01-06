import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from datetime import datetime
import copy
import torch
from filters import butter_lowpass, ideal_lowpass
import torchaudio


class DE:

    def __init__(self, objective_function, pop_fun, X, y, pop_size=50,F=0.5, cr=0.5, use_cuda=False,
                 lowpass='butter', cutoff_freq=0.675, sample_rate=8, SNRdb=10):
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

    def NN_obj(self, agent):
        NN_tx, NN_rx = agent
        Tx = NN_tx(self.X)

        if self.lowpass == 'butter':
            Tx_low = torchaudio.functional.filtfilt(Tx, self.a, self.b)
        elif self.lowpass == 'ideal':
            Tx_freq = torch.fft.rfft(Tx)
            xf = torch.fft.rfftfreq(Tx.shape[2], 1 / self.sample_rate)
            Tx_freq[0][0][xf > self.cutoff] = 0
            Tx_low = torch.fft.irfft(Tx_freq, n=Tx.shape[2])

        Tx_low = Tx_low / torch.sqrt(torch.mean(torch.square(Tx_low)))  # normalize
        Tx_low = Tx_low + torch.normal(0.0, self.sigma, Tx_low.shape, device=self.device)
        received = NN_rx(Tx_low)[0].T

        return self.obj(received, self.y)

    def mutation(self, nets):

        for testp, p1, p2, p3 in zip(*[net.parameters() for net in nets]):
            testp.data = p1 + self.F * (p2 - p3)

        pass

    def crossover(self, target):

        for dw, tw in zip(self.testNN.parameters(), target.parameters()):
            crit = torch.rand(dw.shape, device=self.device) < self.cr
            trial_w = crit * dw + ~crit * tw
            dw.data = trial_w

        pass

    def evolution(self, num_epochs, verbose=False, print_epoch=1000):
        # evaluate the initialized population with the objective function. Only RX agent is evaluated
        obj_all = torch.Tensor([self.NN_obj(agent) for agent in self.pop])

        # find the best agent within the initial population
        self.best_agent = self.pop[torch.argmin(obj_all)]

        best_obj = torch.min(obj_all)
        prev_obj = best_obj

        self.best_objs = np.zeros(num_epochs + 1)
        self.best_objs[0] = best_obj

        for i in range(num_epochs):
            for j, (x_tx, x_rx) in enumerate(self.pop_rx):

                choice = np.random.choice(np.delete(np.arange(self.N), j), 3, replace=False)
                a, b, c = itemgetter(*choice)(self.pop)

                # Mutation
                self.mutation([self.testNN_tx, a[0], b[0], c[0]]) # for Tx
                self.mutation([self.testNN_rx, a[1], b[1], c[1]]) # for Rx

                # Crossover
                self.crossover(x_tx)
                self.crossover(x_rx)

                # Selection
                obj_u = self.NN_obj(self.testNN)
                if obj_u < self.NN_obj(x):
                    self.pop_tx[j] = copy.deepcopy(self.testNN)
                    obj_all[j] = obj_u

            # update the current best objective function value
            best_obj_tx = torch.min(obj_all)
            self.best_objs[i + 1] = best_obj

            if best_obj < prev_obj:
                # update best agent
                self.best_agent = self.pop_tx[torch.argmin(obj_all)]
                # update previous solution to use for next iteration
                prev_obj = best_obj


            if verbose and i % print_epoch == 0:
                # report progress at each iteration
                print('%d: cost= %.5f' % (i, best_obj))
                print('%d: acc= %.5f' % (i, self.accuracy(self.best_agent[1](self.X), self.y)))
                plt.plot(list(self.best_agent[0].parameters())[0].cpu().detach()[0][0]) # Tx network parameters
                plt.plot(list(self.best_agent[1].parameters())[0].cpu().detach()[0][0]) # Rx network parameters
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

        print(f"Best agent is {agent} with a train cost of {np.round(self.NN_obj(agent[1]).cpu().detach(), 5)}.")

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






