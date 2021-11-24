import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from datetime import datetime
import copy
import torch


class DE:

    def __init__(self, objective_function, population_function, X, y, Xtest=None, ytest=None, pop_size=50,
                 F=0.5, cr=0.5, start_agent=None):
        self.obj = objective_function
        self.X = X
        self.y = y
        self.N = pop_size
        self.F = F
        self.cr = cr
        # self.pop = population_function(pop_size)
        self.pop = [population_function() for i in range(pop_size)]
        self.testNN = population_function()
        self.testcost = False
        if start_agent is not None:
            self.pop[0] = copy.deepcopy(start_agent)
        if Xtest is not None and ytest is not None:
            self.Xtest = Xtest
            self.ytest = ytest
            self.testcost = True

    def NN_obj(self, agent):
        yhat = agent(self.X)[0].T
        return self.obj(yhat, self.y.long())

    def mutation(self, nets):

        nets.insert(0, self.testNN)
        for testp, p1, p2, p3 in zip(*[net.parameters() for net in nets]):
            testp.data = p1 + self.F * (p2 - p3)

        pass

    def crossover(self, target):

        for dw, tw in zip(self.testNN.parameters(), target.parameters()):
            r = torch.rand(dw.shape)
            crit = r < self.cr
            trial_w = crit * dw + ~crit * tw
            dw.data = trial_w

        pass

    def evolution(self, num_epochs, verbose=False, print_epoch=1000):
        # evaluate the initialized population with the objective function
        obj_all = [self.NN_obj(agent) for agent in self.pop]

        # find the best agent within the initial population
        self.best_agent = self.pop[np.argmin(obj_all)]

        best_obj = min(obj_all)
        prev_obj = best_obj

        self.best_objs = np.zeros(num_epochs + 1)
        self.best_objs[0] = best_obj

        if self.testcost:
            self.best_test_objs = np.zeros(num_epochs + 1)
            self.best_test_objs[0] = self.obj(self.best_agent(self.Xtest)[0].T, self.ytest.long())

        for i in range(num_epochs):
            for j in range(self.N):

                # Random sampling from the set of all agents exluding the current one, j
                x = self.pop[j]
                choice = np.random.choice(np.delete(np.arange(self.N), j), 3, replace=False)
                a, b, c = itemgetter(*choice)(self.pop)
                # a, b, c = self.pop[np.random.choice(np.delete(np.arange(self.N), j), 3, replace=False)]

                # Mutation
                self.mutation([a, b, c])

                # Crossover
                self.crossover(x)

                # Selection
                obj_u = self.NN_obj(self.testNN)
                if obj_u < self.NN_obj(x):
                    self.pop[j] = copy.deepcopy(self.testNN)
                    obj_all[j] = obj_u

            # update the current best objective function value
            best_obj = min(obj_all)
            self.best_objs[i + 1] = best_obj

            if best_obj < prev_obj:
                # update best agent
                self.best_agent = self.pop[np.argmin(obj_all)]
                # update previous solution to use for next iteration
                prev_obj = best_obj

            if self.testcost:
                self.best_test_objs[i + 1] = self.obj(self.best_agent(self.Xtest)[0].T, self.ytest.long())

            if verbose and i % print_epoch == 0:
                # report progress at each iteration
                print('%d: cost= %.5f' % (i, best_obj))
                print('%d: testcost= %.5f' % (i, self.best_test_objs[i + 1]))
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

        print(f"Best agent is {agent} with a train cost of {np.round(self.NN_obj(agent).detach(), 5)}.")
        print(f"And a test cost of {np.round(self.obj(agent(self.Xtest)[0].T, self.ytest.long()).detach(), 5)}")

        # print(f"Worst initialization was {self.initial_worst_agent} with a cost of \
        # {np.round(self.obj(self.initial_worst_agent), 2)}.")

        pass





