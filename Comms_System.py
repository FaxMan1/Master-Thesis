import numpy as np
import matplotlib.pyplot as plt


class Comms_System:

    def __init__(self, symbol_set, symbol_seq, num_samples=8):
        self.symbol_set = symbol_set
        self.symbol_seq = symbol_seq
        self.m = num_samples
        self.h = self.rrcos()
        self.filter_offset = len(self.h) // 2
        self.start_sample_point = self.filter_offset * 2
        self.new_values, self.new_values_inv = self.filter_calibration()
        pass

    def upsample(self, v=False):
        zero_pads = np.zeros((self.m - 1, len(self.symbol_seq)))
        upsampled = np.vstack((self.symbol_seq, zero_pads)).T.flatten()
        if v:
            plt.figure(figsize=(13, 8))
            plt.stem(upsampled)
            plt.ylabel('Symbol Value', fontsize=18)
            plt.xlabel('Timesteps', fontsize=18)
            plt.title('Upsampled Signal', fontsize=24)
            plt.show()

        return upsampled

    def rrcos(self, beta=0.35, v=False):

        Ts = self.m  # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8 # Ts is symbolperiod
        t = np.arange(-4 * Ts + 1, 4 * Ts)  # remember it's not inclusive of final number
        h = np.sinc(t / Ts) * np.cos(np.pi * beta * t / Ts) / (1 - (2 * beta * t / Ts) ** 2)
        if v:
            plt.figure(figsize=(13, 8))
            plt.plot(t, h)
            plt.title('Root Raised Cosine Filter', fontsize=24)
            plt.show()

        return h

    def plot_filtered(self, filtered_signal, title='Filtered Signal'):
        plt.figure(figsize=(13, 8))
        plt.title(title, fontsize=24)
        plt.plot(filtered_signal, '.-')
        plt.grid(True)
        plt.show()

    def filter_calibration(self, noise_level=None):

        zero_pads = np.zeros((self.m - 1, len(self.symbol_set)))
        upsampled = np.vstack((self.symbol_set, zero_pads)).T.flatten()

        filtered = np.convolve(upsampled, self.h)
        if noise_level is not None:
            filtered = filtered + np.random.normal(0.0, noise_level, filtered.shape)  # add gaussian noise
        double_filtered = np.convolve(filtered, self.h)

        #filter_offset = len(self.h) // 2
        #start_sample_point = filter_offset * 2

        attenuation_factors = {}
        new_values = {}

        for i, s in enumerate(self.symbol_set):
            attenuation_factors[s] = double_filtered[self.start_sample_point + i * self.m] / filtered[self.filter_offset + i * self.m]
            new_values[s] = double_filtered[self.start_sample_point + i * self.m]

        new_values_inv = {v: k for k, v in new_values.items()}
        attenuation_factors_inv = {v: k for k, v in attenuation_factors.items()}

        return new_values, new_values_inv

    def downsample(self, Rx):
        downsampled = np.zeros(len(self.symbol_seq))
        for i in range(len(self.symbol_seq)):
            downsampled[i] = Rx[self.start_sample_point + i * self.m]

        return downsampled

    def decision_making(self, downsampled, v=False):
        chosen_symbols = np.zeros(len(downsampled))
        for i in range(len(downsampled)):
            dists = {}
            for s in self.symbol_set:
                dists[s] = np.linalg.norm(downsampled[i] - s)
            chosen_symbols[i] = min(dists, key=dists.get)
            if v:
                print(dists)

        return chosen_symbols


    def test_CS(self, noise_level=2):

        # calibrate
        #new_values, new_values_inv = self.filter_calibration()

        attenuation_factor = np.max(np.convolve(self.h, self.h))

        # upsample symbol sequence and filter it on transmission side
        upsampled = self.upsample(v=True)
        Tx = np.convolve(upsampled, self.h)
        self.plot_filtered(Tx)

        # Transmit the filtered signal (i.e. add noise)
        Tx = Tx + np.random.normal(0.0, noise_level, Tx.shape)  # add gaussian noise
        self.plot_filtered(Tx, title='Filtered Signal with Noise')

        # Filter on receiver side
        Rx = np.convolve(Tx, self.h)
        self.plot_filtered(Rx, title='Double Filtered')

        # Downsample the signal on the receiver side
        downsampled = self.downsample(Rx)

        # Decision-making using new_values
        decisions = self.decision_making(downsampled/attenuation_factor, False)

        return decisions

    def evaluate(self, decisions):

        #decoded = [self.new_values_inv[dec] for dec in decisions]
        num_errors = np.sum(np.array(decisions) != np.array(self.symbol_seq))
        error_rate = num_errors / len(self.symbol_seq)
        print(decisions[:10], '...')
        print(self.symbol_seq[:10], '...')
        print('{}% error rate'.format(error_rate.round(4)*100))

        return num_errors, error_rate



