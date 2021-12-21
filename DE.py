import numpy as np
import matplotlib.pyplot as plt
from Network import NeuralNetwork
from datetime import datetime
import copy


class DE:

    def __init__(self, objective_function, sizes, X, y, Xtest=None, ytest=None, start_agent=None, pop_size=50,
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
        if Xtest is not None and ytest is not None:
            self.Xtest = Xtest
            self.ytest = ytest
            self.testcost = True

        pass

    def save_params(self, agent=None, wfname='weights', bfname='biases'):
        if agent is None:
            agent = self.best_agent
        d = datetime.now(tz=None)
        date = '_' + str(d.day) + '.' + str(d.month) + '_' + str(d.hour) + '.' + str(d.minute)
        np.savez(wfname + date, *agent.weights)
        np.savez(bfname + date, *agent.biases)

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

        if self.testcost:
            self.best_test_objs = np.zeros(num_epochs + 1)
            self.best_test_objs[0] = self.obj([self.ytest, self.best_agent.feedforward(self.Xtest)])

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

            if self.testcost:
                self.best_test_objs[i + 1] = self.obj([self.ytest, self.best_agent.feedforward(self.Xtest)])

            if verbose and i % print_epoch == 0:
                # report progress at each iteration
                print('%d: cost= %.5f' % (i, best_obj))
                print('%d: testcost= %.5f' % (i, self.best_test_objs[i + 1]))
                print()

        return self.best_agent

    def evaluate(self, plot_function=None, agent=None, bounds=None, savefig=False, title=' '):

        if agent is None:
            agent = self.best_agent
            plt.figure(figsize=(13, 8))
            plt.plot(range(len(self.best_objs)), self.best_objs)
            plt.plot(range(len(self.best_test_objs)), self.best_test_objs)
            plt.title('Training Graph', fontsize=24)
            plt.xlabel('Iterations', fontsize=20)
            plt.ylabel('Cost', fontsize=20)
            plt.legend(['Train', 'Test'], fontsize=14)
            if savefig:
                plt.savefig('TrainingGraph', bbox_inches='tight', transparent=True)
            else:
                plt.show()

        if plot_function is not None:
            plot_function(agent, self.Xtest, self.ytest, title=title, savefig=savefig)

        print(f"Best agent is {agent} with a train cost of {np.round(self.NN_obj(agent), 5)}.")
        print(f"And a test cost of {np.round(self.obj([self.ytest, agent.feedforward(self.Xtest)]), 5)}")

        # print(f"Worst initialization was {self.initial_worst_agent} with a cost of \
        # {np.round(self.obj(self.initial_worst_agent), 2)}.")

        pass

    def accuracy(self, predictions, ytest):
        predictions = predictions.argmax(axis=1)
        true = ytest.argmax(axis=1)
        correct_preds = np.equal(true, predictions)
        return (sum(correct_preds) / len(true))


    def early_stop_training(self, patience, measure='cost', eval=True, v=True):

        n = 1
        iterations = 0
        if measure == 'cost':
            no_iterations_rising = 0
            val_error = 20000
            self.opt_agent = copy.deepcopy(self.pop[np.argmin([self.NN_obj(agent) for agent in self.pop])])
            opt_iterations = iterations
            testcosts = []

            while (no_iterations_rising < patience):
                self.evolution(num_epochs=n, verbose=False, print_epoch=1)
                iterations = iterations + n
                val_error_new = self.obj([self.ytest, self.best_agent.feedforward(self.Xtest)])
                testcosts.append(val_error_new)
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
            self.opt_agent = copy.deepcopy(self.pop[np.argmin([self.NN_obj(agent) for agent in self.pop])])

            while (no_iterations_falling < patience):
                self.evolution(num_epochs=n, verbose=False, print_epoch=1)
                iterations = iterations + n
                val_acc_new = self.accuracy(self.best_agent.feedforward(self.Xtest), self.ytest)
                testcosts.append(val_acc_new)
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
            plt.plot(range(len(testcosts)), testcosts)
            plt.title('Test Cost Graph', fontsize=24)
            plt.xlabel('Iterations', fontsize=20)
            plt.ylabel('Test Cost', fontsize=20)
            plt.show()

        return self.best_agent, self.opt_agent
