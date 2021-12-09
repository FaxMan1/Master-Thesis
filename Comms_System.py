import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter
from ML_components import ML_decision_making, ML_downsampling
from ML_components import network_receiver, network_sender_receiver
from scipy.stats import norm
from scipy.signal import lfilter
from filters import butter_lowpass

class Comms_System:

    def __init__(self, symbol_set, symbol_seq, num_samples=8, beta=0.35):
        self.symbol_set = symbol_set
        self.symbol_seq = symbol_seq
        self.m = num_samples
        self.h = self.rrcos(beta=beta)
        #if norm_h: self.h = self.h / np.sqrt(np.sum(np.square(self.h)))  # normalize the filter
        self.filter_offset = len(self.h) // 2
        self.start_sample_point = self.filter_offset * 2
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


    def plot_spectrum(self, signal_time, sample_rate):
        return plt.magnitude_spectrum(signal_time, Fs=sample_rate, color='C1')


    def rrcos(self, beta, v=False):
        Ts_ = self.m
        t = np.arange(-4 * Ts_, 4 * Ts_ + 1)  # remember it's not inclusive of final number
        h = rrcosfilter(N=8 * Ts_, alpha=beta, Ts=1, Fs=Ts_)[1]
        if v:
            plt.figure(figsize=(13, 8))
            plt.plot(t, h)
            plt.title('Root Raised Cosine Filter', fontsize=24)
            plt.show()

        return h

    def SNRdb_to_sigma(self, SNRdb, energy=None, use_gain=False):
        if energy is None:
            energy = np.mean(np.array(self.symbol_seq) ** 2)
        SNR = 10 ** (SNRdb / 10)
        if use_gain:
            gain_factor = np.max(np.convolve(self.h, self.h))
            sigma = np.sqrt(energy * gain_factor / SNR)  # * gain_factor
        else:
            sigma = np.sqrt(energy / SNR)  # * gain_factor
        return sigma

    def sigma_to_SNRdb(self, sigma):

        avg_symbol_energy = np.mean(np.array(self.symbol_seq) ** 2)
        gain_factor = np.max(np.convolve(self.h, self.h))
        SNR = avg_symbol_energy / (sigma ** 2)
        SNR_with_gain = (avg_symbol_energy * gain_factor) / (sigma ** 2)
        SNRdb = 10 * np.log10(SNR)
        SNRdb_with_gain = 10 * np.log10(SNR_with_gain)

        return SNRdb, SNRdb_with_gain

    def downsample(self, Rx):
        downsampled = np.zeros(len(self.symbol_seq))
        for i in range(len(self.symbol_seq)):
            downsampled[i] = Rx[self.start_sample_point + i * self.m]

        return downsampled

    def get_periods(self, Rx):
        blocks = []
        start = self.start_sample_point - (self.m//2)

        for i in range(len(self.symbol_seq)):
            blocks.append(Rx[start+i*self.m: start + (i+1)*self.m])
        return blocks

    def get_signal_in_blocks(self, Rx):
        blocks = []
        jump = self.start_sample_point

        for i in range(len(self.symbol_seq)):
            blocks.append(Rx[i*self.m: (i*self.m)+jump])

        return blocks

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


    def transmission(self, SNRdb, mode='euclidean', model=None, joint_cutoff=2, rx_cutoff=None, v=False):

        gain_factor = np.max(np.convolve(self.h, self.h))
        upsampled = self.upsample()
        Tx = np.convolve(upsampled, self.h)

        if mode == 'joint':
            sigma = self.SNRdb_to_sigma(SNRdb, 8, use_gain=False)
            if v: print(sigma)
            decisions = network_sender_receiver(upsampled, self.symbol_set, sigma, cutoff_freq=joint_cutoff, models=model)
            return decisions

        if mode == 'network':
            sigma = self.SNRdb_to_sigma(SNRdb, 8, use_gain=False)
            if v: print(sigma)
            if rx_cutoff is not None:
                b, a = butter_lowpass(rx_cutoff, self.m, 10)
                Tx = lfilter(b, a, Tx)
            Tx = Tx / np.sqrt(np.mean(np.square(Tx)))  # normalize signal
            Tx = Tx + np.random.normal(0.0, sigma, Tx.shape)
            decisions = network_receiver(Tx, self.symbol_set, model=model)
            return decisions

        sigma = self.SNRdb_to_sigma(SNRdb, np.mean(self.symbol_seq**2), use_gain=True)
        if v: print(sigma)
        Tx = Tx + np.random.normal(0.0, sigma, Tx.shape)
        Rx = np.convolve(Tx, self.h)
        downsampled = self.downsample(Rx)/gain_factor
        decisions = self.decision_making(downsampled)

        if mode == 'euclidean':
            return decisions

        elif mode == 'NN_decision_making':
            NN_decisions = ML_decision_making(downsampled, self.symbol_set)
            return NN_decisions

        elif mode == 'blocks':
            blocks = self.get_periods(Rx/gain_factor)
            block_decisions = ML_downsampling(blocks, self.symbol_set)
            return block_decisions


        return decisions


    def transmit_all(self, SNRdb, rx_model=None, joint_models=None, joint_cutoff=2, rx_cutoff=None):

        # fordi vi har normaliseret er sample energi sat til 1, og vi har 8 samples pr symbol,
        # altså er avg_symbol_energy så 1*8 for det normaliserede netværk (normaliserede signal)
        #sigma_euclid = self.SNRdb_to_sigma(SNRdb, np.mean(self.symbol_seq**2), use_gain=True)  # symbol energy og gain
        #sigma_network = self.SNRdb_to_sigma(SNRdb, 8, use_gain=False)

        euclid_decisions = self.transmission(SNRdb, mode='euclidean')
        NN_decisions = self.transmission(SNRdb, mode='NN_decision_making')
        block_decisions = self.transmission(SNRdb, mode='blocks')
        network_decisions = self.transmission(SNRdb, mode='network', model=rx_model, rx_cutoff=rx_cutoff)
        joint_decisions = self.transmission(SNRdb, mode='joint', joint_cutoff=joint_cutoff, model=joint_models)

        return euclid_decisions, NN_decisions, block_decisions, network_decisions, joint_decisions


    def evaluate(self, decisions):

        num_errors = np.sum(np.array(decisions) != np.array(self.symbol_seq))
        error_rate = num_errors / len(self.symbol_seq)
        # print('{}% error rate'.format(np.round(error_rate*100, 4)))
        return num_errors, error_rate


def SNR_plot(num_symbols=10000, rx_model=None, joint_models=None, joint_cutoff=2, rx_cutoff=None, all_components=False):
    symbol_set = [3, 1, -1, -3]  # all symbols that we use
    symbol_seq = np.random.choice(symbol_set, num_symbols, replace=True)
    CS = Comms_System(symbol_set=symbol_set, symbol_seq=symbol_seq, num_samples=8, beta=0.35)
    SNRdbs = np.linspace(0, 18, 50)

    euclid_error_rates = np.zeros(len(SNRdbs))
    NN_error_rates = np.zeros(len(SNRdbs))
    block_error_rates = np.zeros(len(SNRdbs))
    network_error_rates = np.zeros(len(SNRdbs))
    joint_error_rates = np.zeros(len(SNRdbs))

    avg_symbol_energy = np.mean(symbol_seq ** 2)
    print('Avg symbol energy', avg_symbol_energy)
    gain_factor = np.max(np.convolve(CS.h, CS.h))
    print('gain', gain_factor)

    # fordi vi har normaliseret er sample energi sat til 1, og vi har 8 samples pr symbol,
    # altså er avg_symbol_energy så 1*8 for det normaliserede netværk (normaliserede signal)

    for i, SNRdb in enumerate(SNRdbs):

        #sigma_euclid = CS.SNRdb_to_sigma(SNRdb, avg_symbol_energy, use_gain=True)  # symbol energy og gain
        #sigma_network = CS.SNRdb_to_sigma(SNRdb, 8, use_gain=False)

        euclid_decisions = CS.transmission(SNRdb, mode='euclidean')
        network_decisions = CS.transmission(SNRdb, mode='network', model=rx_model, rx_cutoff=rx_cutoff)
        joint_decisions = CS.transmission(SNRdb, mode='joint', joint_cutoff=joint_cutoff, model=joint_models)
        if all_components:
            NN_decisions = CS.transmission(SNRdb, mode='NN_decision_making')
            block_decisions = CS.transmission(SNRdb, mode='blocks')
            NN_error_rates[i] = CS.evaluate(NN_decisions)[1]
            block_error_rates[i] = CS.evaluate(block_decisions)[1]

        euclid_error_rates[i] = CS.evaluate(euclid_decisions)[1]
        network_error_rates[i] = CS.evaluate(network_decisions)[1]
        joint_error_rates[i] = CS.evaluate(joint_decisions)[1]

    sigmas = np.array([CS.SNRdb_to_sigma(SNRdb, avg_symbol_energy, use_gain=True) for SNRdb in SNRdbs])
    error_theory = 1.5 * (1 - norm.cdf(np.sqrt(gain_factor / sigmas ** 2)))

    return SNRdbs, euclid_error_rates, network_error_rates, NN_error_rates, block_error_rates, joint_error_rates, error_theory







