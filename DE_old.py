import numpy as np
import matplotlib.pyplot as plt


# Differential Evolution on vectors (Normal differential evolution) (Rand/1/Bin)
class DE():

    def __init__(self, objective_function, bounds, pop_size, num_dimensions, F=0.5, cr=0.5):
        self.obj = objective_function
        self.p = num_dimensions
        self.bounds = np.array([bounds, ] * self.p)
        self.N = pop_size
        self.pop = np.random.uniform(bounds[0], bounds[1], (self.N, len(self.bounds)))
        self.F = F
        self.cr = cr

        pass

    def mutation(self, t):

        return t[0] + self.F * (t[1] - t[2])

    # binomial crossover scheme
    def crossover(self, donor, target):

        r = np.random.rand(self.p)
        return np.array([donor[i] if r[i] < self.cr else target[i] for i in range(self.p)])

    def evaluate(self, plot_function=None, bounds=None, savefig=False):

        plt.figure(figsize=(13, 8))
        plt.plot(range(len(self.best_objs)), self.best_objs)
        plt.title('Training Graph', fontsize=24)
        plt.xlabel('Iterations', fontsize=20)
        plt.ylabel('Fitness Value', fontsize=20)
        if savefig:
            plt.savefig('TrainingGraph', bbox_inches='tight', transparent=True)
        else:
            plt.show()

        print(f"Worst initialization was {self.initial_worst_agent} with a cost of \
        {np.round(self.obj(self.initial_worst_agent), 2)}.")

        print(f"Best agent is {self.best_agent} with a cost of \
              {np.round(self.obj(self.best_agent), 30)}.")

        if plot_function is not None:
            plot_function(self.obj, self.initial_worst_agent, bounds, 'Worst Initialization', savefig=savefig)
            plot_function(self.obj, self.best_agent, bounds, 'Best Solution', savefig=savefig)

        pass

    def evolution(self, num_epochs, verbose=False):
        # evaluate the initialized population with the objective function
        obj_all = [self.obj(agent) for agent in self.pop]

        # find the best agent within the initial population
        self.best_agent = self.pop[np.argmin(obj_all)]
        self.initial_worst_agent = self.pop[np.argmax(obj_all)].copy()
        best_obj = min(obj_all)
        prev_obj = best_obj
        self.best_objs = np.zeros(num_epochs + 1)
        self.best_objs[0] = best_obj

        for i in range(num_epochs):
            for j in range(self.N):
                # random samples from the set of all agents exluding the current one, j
                x = self.pop[j]
                a, b, c = self.pop[np.random.choice(np.delete(np.arange(self.N), j), 3, replace=False)]
                v = self.mutation([a, b, c])
                # if bounds are the same for all dimensions, one could simply use "np.clip(donor, lb, ub)""
                v = [np.clip(v[k], lb, ub) for k, (lb, ub) in
                     enumerate(self.bounds)]  # maybe throw to separate function?
                u = self.crossover(v, x)
                # selection
                if self.obj(u) < self.obj(x):
                    self.pop[j] = u
                    obj_all[j] = self.obj(u)  # maybe minimize number of function calls to improve runtime?

            # update the current best objective function value
            best_obj = min(obj_all)
            self.best_objs[i + 1] = best_obj

            if best_obj < prev_obj:
                # update best agent
                self.best_agent = self.pop[np.argmin(obj_all)]
                # update previous solution to use for next iteration
                prev_obj = best_obj

            if verbose:
                # report progress at each iteration
                print('%d: Agent= %s, cost= %.5f' % (i, np.round(self.best_agent, decimals=5), best_obj))

        return self.best_agent
