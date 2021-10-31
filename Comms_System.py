import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter


class Comms_System:

    def __init__(self, symbol_set, symbol_seq, num_samples=8, beta=0.35):
        self.symbol_set = symbol_set
        self.symbol_seq = symbol_seq
        self.m = num_samples
        self.h = self.rrcos2(beta=beta)
        self.filter_offset = len(self.h) // 2
        self.start_sample_point = self.filter_offset * 2
        # self.new_values, self.new_values_inv = self.filter_calibration()
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

    def rrcos2(self, beta, v=False):
        Ts_ = self.m
        t = np.arange(-4 * Ts_, 4 * Ts_ + 1)  # remember it's not inclusive of final number
        h = rrcosfilter(N=8 * Ts_, alpha=beta, Ts=1, Fs=Ts_)[1]
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

    def test_CS(self, noise_level=2, v=False):

        # calibrate
        gain_factor = np.max(np.convolve(self.h, self.h))

        # upsample symbol sequence and filter it on transmission side
        upsampled = self.upsample(v=v)
        Tx = np.convolve(upsampled, self.h)

        if v:
            self.plot_filtered(Tx)

        # Transmit the filtered signal (i.e. add noise)
        Tx = Tx + np.random.normal(0.0, noise_level, Tx.shape)  # add gaussian noise

        # Filter on receiver side
        Rx = np.convolve(Tx, self.h)

        if v:
            self.plot_filtered(Tx, title='Filtered Signal with Noise')
            self.plot_filtered(Rx, title='Received Signal')

        # Downsample the signal on the receiver side
        downsampled = self.downsample(Rx)

        # Decision-making using new_values
        decisions = self.decision_making(downsampled/gain_factor, False)

        return decisions

    def evaluate(self, decisions):

        num_errors = np.sum(np.array(decisions) != np.array(self.symbol_seq))
        error_rate = num_errors / len(self.symbol_seq)
        # print('{}% error rate'.format(np.round(error_rate*100, 4)))

        return num_errors, error_rate



