import numpy as np
import matplotlib.pyplot as plt
from Network import NeuralNetwork
from datetime import datetime
import copy


class DE:

    def __init__(self, objective_function, sizes, X, y, start_agent=None, pop_size=50, F=0.5, cr=0.5, softmax=False):
        self.obj = objective_function
        self.sizes = sizes
        self.X = X
        self.y = y
        self.N = pop_size
        self.F = F
        self.cr = cr
        self.pop = np.array([NeuralNetwork(sizes, softmax=softmax) for i in range(self.N)])
        if start_agent is not None:
            self.pop[0] = start_agent
        self.testNN = NeuralNetwork(sizes, softmax=softmax)

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

    def NN_obj(self, agent, chosen):

        yhat = agent.feedforward(self.X[chosen])
        return self.obj([self.y[chosen], yhat])

    def mutation(self, nets):

        for i in range(
                len(nets[0].weights)):  # iterate over all weights. t[0].weights, t[1].weights etc are same length
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

    def evolve(self, j, chosen):

        # Random sampling from the set of all agents exluding the current one, j
        x = self.pop[j]
        a, b, c = self.pop[np.random.choice(np.delete(np.arange(self.N), j), 3, replace=False)]

        # Mutation
        self.mutation([a, b, c])

        # Crossover
        self.crossover(x)

        # Selection
        obj_u = self.NN_obj(self.testNN, chosen)
        if obj_u < self.NN_obj(x, chosen):
            self.pop[j] = copy.deepcopy(self.testNN)
            self.obj_all[j] = obj_u

    def evolution(self, num_epochs, batch_size, verbose=False, print_epoch=1000):
        idx = np.arange(self.X.shape[0])
        iterations_per_epoch = self.X.shape[0] // batch_size
        chosen1 = np.random.choice(idx, batch_size, replace=False)

        # evaluate the initialized population with the objective function
        self.obj_all = [self.NN_obj(agent, chosen1) for agent in self.pop]

        # find the best agent within the initial population
        self.best_agent = self.pop[np.argmin(self.obj_all)]
        # self.initial_worst_agent = self.pop[np.argmax(obj_all)].copy()
        best_obj = min(self.obj_all)
        prev_obj = best_obj
        self.best_objs = np.zeros(num_epochs + 1)
        self.best_objs[0] = best_obj

        # iterate through every epoch
        for i in range(num_epochs):
            # iterate through all agents in the population
            for k in range(iterations_per_epoch):
                chosen = np.random.choice(idx, batch_size, replace=False)
                for j in range(self.N):
                    self.evolve(j, chosen)  # evolve the agent

                # after all agents in the minibatch have been evolved, update the current best objective function value
                best_obj = min(self.obj_all)
                self.best_objs[i + 1] = best_obj

                if best_obj < prev_obj:
                    # update best agent
                    self.best_agent = self.pop[np.argmin(self.obj_all)]
                    # update previous solution to use for next iteration
                    prev_obj = best_obj

            if verbose and i % print_epoch == 0:
                # report progress at each iteration
                print('%d: cost= %.5f' % (i, best_obj))

        return self.best_agent

    def evaluate(self, xtest, plot_function=None, agent=None, bounds=None, title=' '):

        if agent is None:
            agent = self.best_agent
            plt.figure(figsize=(13, 8))
            plt.plot(range(len(self.best_objs)), self.best_objs)
            plt.title('Training Graph', fontsize=24)
            plt.xlabel('Iterations', fontsize=20)
            plt.ylabel('Fitness Value', fontsize=20)
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

        print(f"Best agent is {agent} with a total cost of \
              {np.round(self.NN_obj(agent, np.arange(self.X.shape[0])), 30)}.")

        # print(f"Worst initialization was {self.initial_worst_agent} with a cost of \
        # {np.round(self.obj(self.initial_worst_agent), 2)}.")

        pass