import numpy as np
import matplotlib.pyplot as plt
from Network import NeuralNetwork
from datetime import datetime
import copy


class DE:

    def __init__(self, objective_function, sizes, X, y, start_agent=None, pop_size=50,
                 F=0.5, cr=0.5, type='classification', afunc='tanh', softmax=True):
        self.obj = objective_function
        self.sizes = sizes
        self.X = X
        self.y = y
        self.N = pop_size
        self.F = F
        self.cr = cr
        self.pop = np.array([NeuralNetwork(sizes, type=type, afunc=afunc, softmax=softmax) for i in range(self.N)])
        if start_agent is not None:
            self.pop[0] = start_agent
        self.testNN = NeuralNetwork(sizes, type=type, afunc=afunc, softmax=softmax)

        pass

    def save_params(self, wfname='weights', bfname='biases'):
        d = datetime.now(tz=None)
        date = '_' + str(d.day) + '.' + str(d.month) + '_' + str(d.hour) + '.' + str(d.minute)
        np.savez(wfname + date, *self.best_agent.weights)
        np.savez(bfname + date, *self.best_agent.biases)

    def load_params(self, weight_file, bias_file):

        w_container = np.load(weight_file)
        b_container = np.load(bias_file)
        weights = [w_container[key] for key in w_container]
        biases = [b_container[key] for key in b_container]

        return weights, biases

    def NN_obj(self, agent):

        yhat = agent.feedforward(self.X)
        return self.obj([self.y, yhat])

    def mutation(self, nets):

        for i in range(len(nets[0].weights)):  # iterate over all weights. t[0].weights, t[1].weights... are same len
            self.testNN.weights[i] = nets[0].weights[i] + self.F * (nets[1].weights[i] - nets[2].weights[i])

        # biases
        for i in range(len(nets[0].biases)):  # iterate over all biases
            self.testNN.biases[i] = nets[0].biases[i] + self.F * (nets[1].biases[i] - nets[2].biases[i])

        pass

    def crossover(self, target):  # crossover_tensor

        # donor is self.testNN, target is x

        # weights
        for i, (dw, tw) in enumerate(zip(self.testNN.weights, target.weights)):
            r = np.random.uniform(0, 1, size=dw.shape)
            crit = r < self.cr
            trial_w = crit * dw + ~crit * tw
            self.testNN.weights[i] = trial_w

        # biases
        for i, (db, tb) in enumerate(zip(self.testNN.biases, target.biases)):
            r = np.random.uniform(0, 1, size=db.shape)
            crit = r < self.cr
            trial_b = crit * db + ~crit * tb
            self.testNN.biases[i] = trial_b

        pass

    def evolution(self, num_epochs, verbose=False, print_epoch=1000):
        # evaluate the initialized population with the objective function
        obj_all = [self.NN_obj(agent) for agent in self.pop]

        # find the best agent within the initial population
        self.best_agent = self.pop[np.argmin(obj_all)]
        # self.initial_worst_agent = self.pop[np.argmax(obj_all)].copy()
        best_obj = min(obj_all)
        prev_obj = best_obj
        self.best_objs = np.zeros(num_epochs + 1)
        self.best_objs[0] = best_obj

        for i in range(num_epochs):
            for j in range(self.N):

                # Random sampling from the set of all agents exluding the current one, j
                x = self.pop[j]
                a, b, c = self.pop[np.random.choice(np.delete(np.arange(self.N), j), 3, replace=False)]

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

            if verbose and i % print_epoch == 0:
                # report progress at each iteration
                print('%d: cost= %.5f' % (i, best_obj))

        return self.best_agent

    def evaluate(self, xtest, ytest, plot_function=None, agent=None, bounds=None, title=' '):

        if agent is None:
            agent = self.best_agent
            plt.figure(figsize=(13, 8))
            plt.plot(range(len(self.best_objs)), self.best_objs)
            plt.title('Training Graph', fontsize=24)
            plt.xlabel('Iterations', fontsize=20)
            plt.ylabel('Fitness Value (MSE)', fontsize=20)
            plt.show()

        if plot_function is not None:

            x1, x2 = xtest[:, 0], xtest[:, 1]
            x1mesh, x2mesh = np.meshgrid(x1, x2)
            yhats = []
            xtest3d = []

            for x1_, x2_ in zip(x1mesh, x2mesh):
                xtest3d.append(np.column_stack([x1_, x2_]))

            for x in xtest3d:
                yhat = agent.feedforward(x)
                yhats.append(yhat.flatten())

            yhats = np.array(yhats)
            plot_function(xtest, yhats, title=title, savefig=False)
            # plot_function(xtest, title='Label Gaussian')

        print(f"Best agent is {agent} with a train cost of {np.round(self.NN_obj(agent), 5)}.")
        print(f"And a test cost of {np.round(self.obj([ytest, agent.feedforward(xtest)]), 5)}")

        # print(f"Worst initialization was {self.initial_worst_agent} with a cost of \
        # {np.round(self.obj(self.initial_worst_agent), 2)}.")

        pass
