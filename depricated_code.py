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


avg_symbol_energy = np.mean(np.array(self.symbol_seq) ** 2)
SNR = (avg_symbol_energy) / (noise_level ** 2)
SNR_with_gain = (avg_symbol_energy * gain_factor) / (noise_level ** 2)
SNRdb = 10*np.log10(SNR)
SNRdb_with_gain = 10*np.log10(SNR_with_gain)
print(SNRdb)
print(SNRdb_with_gain)

#rev_eng_SNR = 10 ** (noise_level/10)
#rev_eng_sigma = np.sqrt((avg_symbol_energy * gain_factor)/rev_eng_SNR)
# noise_level = rev_eng_sigma

#sigma = 0.35 # corresponds roughly to SNR 16 (old sigma=1)
#sigma = 0.7 # corresponds roughly to SNR 10 (old sigma=2)
#sigma = 1.06 # corresponds roughly to SNR 6.4 (old sigma=3)



def transmission(self, mode='euclidean', noise_level=2, norm_signal=False, model=None, v=True):
    # calculate gain_factor
    gain_factor = np.max(np.convolve(self.h, self.h))
    energy = np.mean(np.array(self.symbol_seq) ** 2)
    if v:
        print("E:", gain_factor)
        print('Ratio:', energy * gain_factor / noise_level ** 2)
    # upsample
    upsampled = self.upsample()

    # filter with rrcos
    Tx = np.convolve(upsampled, self.h)

    # normalize
    if norm_signal:
        Tx = Tx / np.sqrt(np.mean(np.square(Tx)))  # np.sqrt(np.mean(np.square(Tx)))

    # Transmit / add noise
    Tx = Tx + np.random.normal(0.0, noise_level, Tx.shape)  # add gaussian noise

    # filter with rrcos on receiver
    if norm_signal:
        Rx = np.convolve(Tx, self.h) * np.sqrt(np.mean(self.symbol_seq ** 2))
    else:
        Rx = np.convolve(Tx, self.h)

    # downsample
    if norm_signal:
        downsampled = self.downsample(Rx)
    else:
        downsampled = self.downsample(Rx) / gain_factor

    if mode == 'euclidean':
        received_symbols = self.decision_making(downsampled, False)
    elif mode == 'network':
        received_symbols = network_receiver(Tx, self.symbol_set, model=model)

    return received_symbols


def SNR_plot(num_symbols=10000, lowpass=None, conv_model=None, norm_h=True, norm_signal=False, use_gain=False):
    symbol_set = [3, 1, -1, -3]  # all symbols that we use
    symbol_seq = np.random.choice(symbol_set, num_symbols, replace=True)
    m = 8
    CS = Comms_System(symbol_set=symbol_set, symbol_seq=symbol_seq, num_samples=m, beta=0.35, norm_h=norm_h)

    #sigmas = np.linspace(CS.SNR_to_sigma(18), CS.SNR_to_sigma(2), 50)  # sigmas = np.linspace(2.5, 4.5, 500)#
    SNRdbs = np.linspace(0, 18, 50)

    sigmas = []
    euclid_error_rates = []
    error_rates_NN = []
    error_rates_NN_blocks = []
    error_rates_NN_filter = []
    error_rates_conv = []
    avg_symbol_energy = np.mean(np.array(symbol_seq) ** 2)
    print('Avg symbol energy', avg_symbol_energy)
    gain_factor = np.max(np.convolve(CS.h, CS.h))
    print('gain', gain_factor)


    #for sigma in sigmas:
    for SNRdb in SNRdbs:
        sigma = CS.SNRdb_to_sigma(SNRdb, avg_symbol_energy, use_gain=use_gain)
        euclid_decisions, NN_decisions, block_decisions, filter_decisions, conv_decisions, _ = CS.test_CS(
            noise_level=sigma, lowpass=lowpass, conv_model=conv_model, norm_signal=norm_signal)

        sigmas.append(sigma)
        euclid_error_rates.append(CS.evaluate(euclid_decisions)[1])
        error_rates_NN.append(CS.evaluate(NN_decisions)[1])
        error_rates_NN_blocks.append(CS.evaluate(block_decisions)[1])
        error_rates_NN_filter.append(CS.evaluate(filter_decisions)[1])
        error_rates_conv.append(CS.evaluate(conv_decisions)[1])

    #SNRsDB = 10 * np.log10(SNRs)
    euclid_error_rates = np.array(euclid_error_rates)
    error_rates_NN = np.array(error_rates_NN)
    error_rates_NN_blocks = np.array(error_rates_NN_blocks)
    error_rates_NN_filter = np.array(error_rates_NN_filter)
    error_rates_conv = np.array(error_rates_conv)
    sigmas = np.array(sigmas)
    error_theory = 1.5 * (1 - norm.cdf(np.sqrt(gain_factor / sigmas ** 2)))  #

    return SNRdbs, euclid_error_rates, error_rates_NN, error_rates_NN_blocks, error_rates_NN_filter, error_rates_conv, error_theory


def test_CS(self, noise_level=2, dec_model=None, block_model=None, filter_model=None,
        conv_model=None, lowpass=None, v=False, norm_signal=False):

    gain_factor = np.max(np.convolve(self.h, self.h))
    gain_factor_tx = np.max(self.h)
    if v:
        print(gain_factor)

    # upsample symbol sequence and filter it on transmission side
    upsampled = self.upsample(v=v)
    if lowpass is not None:
        b, a = butter_lowpass(lowpass, self.m, 4)
        upsampled = signal.lfilter(b, a, upsampled)
    Tx = np.convolve(upsampled, self.h)
    if v:
        self.plot_filtered(Tx)

    if norm_signal:
        # normalize filtered signal before sending
        Tx = Tx / np.sqrt(np.mean(np.square(Tx)))
    # Transmit the filtered signal (i.e. add noise)
    Tx = Tx + np.random.normal(0.0, noise_level, Tx.shape)  # add gaussian noise

    # Filter on receiver side
    Rx = np.convolve(Tx, self.h)
    blocks = self.get_periods(Rx/gain_factor)
    filter_blocks = self.get_signal_in_blocks(Tx/gain_factor_tx)
    if v:
        self.plot_filtered(Tx, title='Filtered Signal with Noise')
        self.plot_filtered(Rx, title='Received Signal')

    # Downsample the signal on the receiver side
    if norm_signal:
        downsampled = self.downsample(Rx)
    else:
        downsampled = self.downsample(Rx)/gain_factor

    # Decision-making downsampled values
    euclid_decisions = self.decision_making(downsampled, False)
    NN_decisions = ML_decision_making(downsampled, self.symbol_set, model=dec_model)
    block_decisions = ML_downsampling(blocks, self.symbol_set, model=block_model)
    filter_decisions = ML_filtering(filter_blocks, self.symbol_set, model=filter_model)
    conv_decisions = network_receiver(Tx, self.symbol_set, model=conv_model)

    return euclid_decisions, NN_decisions, block_decisions, filter_decisions, conv_decisions, downsampled


'''if norm_nets:
            sigma_euclid = CS.SNRdb_to_sigma(SNRdb, avg_symbol_energy, use_gain=True) # symbol energy og gain
            sigma_network = CS.SNRdb_to_sigma(SNRdb, 8, use_gain=False)
            euclid_decisions = CS.transmission(sigma_euclid, mode='euclidean')
            if all_components:
                NN_decisions = CS.transmission(sigma_euclid, mode='NN_decision_making')
                block_decisions = CS.transmission(sigma_euclid, mode='blocks')
            network_decisions = CS.transmission(sigma_network, mode='network', model=rx_model)
            joint_decisions = CS.transmission(sigma_network, mode='joint', cutoff=cutoff, model=joint_models)
            sigmas[i] = sigma_euclid
        else:
            sigma = CS.SNRdb_to_sigma(SNRdb, avg_symbol_energy, use_gain=True)
            euclid_decisions = CS.transmission(sigma)
            network_decisions = CS.transmission_no_norm(sigma, mode='network', model=rx_model)
            sigmas[i] = sigma'''
'''

for i, SNRdb in enumerate(SNRdbs):

    # sigma_euclid = CS.SNRdb_to_sigma(SNRdb, avg_symbol_energy, use_gain=True)  # symbol energy og gain
    # sigma_network = CS.SNRdb_to_sigma(SNRdb, 8, use_gain=False)
    euclid_decisions = CS.transmission(SNRdb, mode='euclidean')

    if norm_nets:
        network_decisions = CS.transmission(SNRdb, mode='network', model=rx_model, rx_cutoff=rx_cutoff)
        joint_decisions = CS.transmission(SNRdb, mode='joint', joint_cutoff=joint_cutoff, model=joint_models)
    else:
        network_decisions = CS.transmission_no_norm(SNRdb, mode='network', model=rx_model)

    if all_components:
        NN_decisions = CS.transmission(SNRdb, mode='NN_decision_making')
        block_decisions = CS.transmission(SNRdb, mode='blocks')

    euclid_error_rates[i] = CS.evaluate(euclid_decisions)[1]
    network_error_rates[i] = CS.evaluate(network_decisions)[1]
    if all_components:
        NN_error_rates[i] = CS.evaluate(NN_decisions)[1]
        block_error_rates[i] = CS.evaluate(block_decisions)[1]
    if norm_nets:
        joint_error_rates[i] = CS.evaluate(joint_decisions)[1]


 def transmission_no_norm(self, SNRdb=10, mode='euclidean', model=None, v=False):

        sigma = self.SNRdb_to_sigma(SNRdb, np.mean(self.symbol_seq**2), use_gain=True)
        if v: print('Sigma:', sigma)

        gain_factor = np.max(np.convolve(self.h, self.h))
        upsampled = self.upsample()
        Tx = np.convolve(upsampled, self.h)
        # If normalize: Tx = Tx / np.sqrt(np.mean(np.square(Tx)))
        Tx = Tx + np.random.normal(0.0, sigma, Tx.shape)
        Rx = np.convolve(Tx, self.h)
        # if normalize: Rx =  (Rx / np.sqrt(np.mean(np.square(Rx)))) *
                            # np.sqrt(np.mean(self.symbol_seq ** 2))

        downsampled = self.downsample(Rx)/gain_factor

        if mode == 'euclidean':
            received_symbols = self.decision_making(downsampled, False)
        elif mode == 'network':
            received_symbols = network_receiver(Tx, self.symbol_set, model=model)

        return received_symbols'''

idx_y = np.arange(self.y.shape[0])
y_chosen = np.random.choice(idx_y.shape[0], 1, replace=False)
y_chosen = np.arange(y_chosen, y_chosen + batch_size)
y_chosen = y_chosen[y_chosen < self.y.shape[0]]
y_start, y_end = y_chosen[0], y_chosen[-1]
x_chosen = np.arange(y_start * 8, (y_end + 1) * 8)
x_chosen = x_chosen[x_chosen < self.X.shape[2]]
idx_y = np.array(
    [x for x in idx_y if x not in y_chosen])  # removes the already selected elements. Corresponds to replace=False
self.evolve(x_chosen, y_chosen)


def get_minibatch(self, idx_y, do_not_replace=True):
    y_chosen = torch.multinomial(idx_y, 1, replacement=False).item()
    y_chosen = torch.arange(y_chosen, y_chosen + 200, device=self.device)
    y_chosen = y_chosen[y_chosen < self.y.shape[0]]
    y_start, y_end = y_chosen[0], y_chosen[-1]
    x_chosen = torch.arange(y_start * 8, (y_end + 1) * 8, device=self.device)
    x_chosen = x_chosen[x_chosen < self.X.shape[2]]
    # print(x_chosen.is_cuda)
    # print(y_chosen.is_cuda)
    if do_not_replace:
        idx_y = torch.tensor([x for x in idx_y if x not in y_chosen], device=self.device)
        return x_chosen, y_chosen, idx_y
    else:
        return x_chosen, y_chosen

