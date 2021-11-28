import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from datetime import datetime
import copy
import torch


class DE:

    def __init__(self, objective_function, population_function, X, y, Xtest=None, ytest=None, pop_size=50,
                 F=0.5, cr=0.5, start_agent=None, use_cuda=False):
        if use_cuda and torch.cuda.is_available:
            self.device = torch.device('cuda')
            print('Using GPU')
        else:
            self.device = torch.device('cpu')
        self.obj = objective_function
        self.X = X.to(self.device)
        self.y = y.long().to(self.device)
        self.N = pop_size
        self.pop_func = population_function
        self.F = torch.Tensor([F]).to(self.device)
        self.cr = torch.Tensor([cr]).to(self.device)
        self.pop = [population_function().to(self.device) for i in range(pop_size)]
        self.testNN = population_function().to(self.device)
        self.testcost = False
        if start_agent is not None:
            self.pop[0] = copy.deepcopy(start_agent)
        if Xtest is not None and ytest is not None:
            self.Xtest = Xtest.to(self.device)
            self.ytest = ytest.long().to(self.device)
            self.testcost = True

    def save_params(self, fname, path='../Conv1DWeights/', agent=None):
        if agent is None:
            agent = self.best_agent
        torch.save(agent.state_dict(), f=path + fname)
        pass

    def save_model(self, fname, path='../Conv1DModels/', agent=None):
        if agent is None:
            agent = self.best_agent
        torch.save(agent, path+fname)

    def load_params(self, fname, path='../Conv1DWeights/'):
        new_NN = self.pop_func()
        new_NN.load_state_dict(torch.load(path + fname))
        return new_NN

    def load_model(self, fname, path='../Conv1DModels/'):
        return torch.load(path+fname)

    def NN_obj(self, agent):
        yhat = agent(self.X)[0].T
        return self.obj(yhat, self.y)

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
        # evaluate the initialized population with the objective function
        obj_all = torch.Tensor([self.NN_obj(agent) for agent in self.pop])

        # find the best agent within the initial population
        self.best_agent = self.pop[torch.argmin(obj_all)]

        best_obj = torch.min(obj_all)
        prev_obj = best_obj

        self.best_objs = np.zeros(num_epochs + 1)
        self.best_objs[0] = best_obj

        if self.testcost:
            self.best_test_objs = np.zeros(num_epochs + 1)
            self.best_test_objs[0] = self.obj(self.best_agent(self.Xtest)[0].T, self.ytest)

        for i in range(num_epochs):
            for j, x in enumerate(self.pop):

                choice = np.random.choice(np.delete(np.arange(self.N), j), 3, replace=False)
                a, b, c = itemgetter(*choice)(self.pop)
                # a, b, c = self.pop[np.random.choice(np.delete(np.arange(self.N), j), 3, replace=False)]

                # Mutation
                self.mutation([self.testNN, a, b, c])

                # Crossover
                self.crossover(x)

                # Selection
                obj_u = self.NN_obj(self.testNN)
                if obj_u < self.NN_obj(x):
                    self.pop[j] = copy.deepcopy(self.testNN)
                    obj_all[j] = obj_u

            # update the current best objective function value
            best_obj = torch.min(obj_all)
            self.best_objs[i + 1] = best_obj

            if best_obj < prev_obj:
                # update best agent
                self.best_agent = self.pop[torch.argmin(obj_all)]
                # update previous solution to use for next iteration
                prev_obj = best_obj

            if self.testcost:
                self.best_test_objs[i + 1] = self.obj(self.best_agent(self.Xtest)[0].T, self.ytest)

            if verbose and i % print_epoch == 0:
                # report progress at each iteration
                print('%d: cost= %.5f' % (i, best_obj))
                print('%d: testcost= %.5f' % (i, self.best_test_objs[i + 1]))
                print('%d: acc= %.5f' % (i, self.accuracy(self.best_agent(self.X), self.y)))
                print('%d: testacc= %.5f' % (i, self.accuracy(self.best_agent(self.Xtest), self.ytest)))
                print()

        return self.best_agent

    def evaluate(self, plot_function=None, agent=None, bounds=None, title=' '):

        if agent is None:
            agent = self.best_agent
            plt.figure(figsize=(13, 8))
            plt.plot(range(len(self.best_objs)), self.best_objs)
            plt.plot(range(len(self.best_test_objs)), self.best_test_objs)
            plt.title('Training Graph', fontsize=24)
            plt.xlabel('Iterations', fontsize=20)
            plt.ylabel('Cost', fontsize=20)
            plt.legend(['Train', 'Test'], fontsize=14)
            plt.show()

        if plot_function is not None:
            plot_function(agent, self.Xtest, self.ytest, title=title, savefig=False)

        print(f"Best agent is {agent} with a train cost of {np.round(self.NN_obj(agent).cpu().detach(), 5)}.")
        print(f"And a test cost of {np.round(self.obj(agent(self.Xtest)[0].T, self.ytest).cpu().detach(), 5)}")

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






