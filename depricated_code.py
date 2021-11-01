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