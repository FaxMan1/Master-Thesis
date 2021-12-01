def filter_calibration(self, noise_level=None):
    zero_pads = np.zeros((self.m - 1, len(self.symbol_set)))
    upsampled = np.vstack((self.symbol_set, zero_pads)).T.flatten()

    filtered = np.convolve(upsampled, self.h)
    if noise_level is not None:
        filtered = filtered + np.random.normal(0.0, noise_level, filtered.shape)  # add gaussian noise
    double_filtered = np.convolve(filtered, self.h)

    # filter_offset = len(self.h) // 2
    # start_sample_point = filter_offset * 2

    attenuation_factors = {}
    new_values = {}

    for i, s in enumerate(self.symbol_set):
        attenuation_factors[s] = double_filtered[self.start_sample_point + i * self.m] / filtered[
            self.filter_offset + i * self.m]
        new_values[s] = double_filtered[self.start_sample_point + i * self.m]

    new_values_inv = {v: k for k, v in new_values.items()}
    attenuation_factors_inv = {v: k for k, v in attenuation_factors.items()}

    return new_values, new_values_inv

def rrcos(self, beta, v=False):

    Ts = self.m  # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8 # Ts is symbolperiod
    t = np.arange(-4 * Ts, 4 * Ts+1)  # remember it's not inclusive of final number
    # h = np.sinc(t / Ts) * np.cos(np.pi * beta * t / Ts) / (1 - (2 * beta * t / Ts) ** 2)
    h = (np.cos((1+beta) * np.pi *t/Ts) + np.pi * (1-beta)/4/beta * np.sinc((1-beta) * t/Ts))/(1-(4 * beta * t/Ts)**2)
    if v:
        plt.figure(figsize=(13, 8))
        plt.plot(t, h)
        plt.title('Root Raised Cosine Filter', fontsize=24)
        plt.show()

    return h


def plot_filtered(self, filtered_signal, title='Filtered Signal', show_sample_points=False):
    plt.figure(figsize=(13, 8))
    plt.title(title, fontsize=24)
    plt.plot(filtered_signal, '.-')
    # if show_sample_points:
        # for i in range(len(self.symbol_seq)):
            # plt.plot([self.start_sample_point+ i * self.m, i * self.m + self.start_sample_point], [min(filtered_signal), max(filtered_signal)], alpha=0.7)
    plt.grid(True)
    plt.show()




# Evolution Without mini-batch

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

    def evaluate(self, plot_function=None, bounds=None):

        plt.figure(figsize=(13, 8))
        plt.plot(range(len(self.best_objs)), self.best_objs)
        plt.title('Training Graph', fontsize=24)
        plt.xlabel('Iterations', fontsize=20)
        plt.ylabel('Fitness Value', fontsize=20)
        plt.show()

        print(f"Worst initialization was {self.initial_worst_agent} with a cost of \
        {np.round(self.obj(self.initial_worst_agent), 2)}.")

        print(f"Best agent is {self.best_agent} with a cost of \
              {np.round(self.obj(self.best_agent), 30)}.")

        if plot_function is not None:
            plot_function(self.obj, self.initial_worst_agent, bounds, 'Worst Initialization')
            plot_function(self.obj, self.best_agent, bounds, 'Best Solution')

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

def spectrum_alt(signal, sample_rate, N):

    yf = scipy.fft.rfft(signal)
    xf = scipy.fft.rfftfreq(N, 1/sample_rate)
    plt.plot(xf, np.abs(yf))
    plt.show()