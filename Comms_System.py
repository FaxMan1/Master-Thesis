import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter
from ML_components import ML_decision_making, ML_downsampling, ML_filtering, network_receiver
from scipy import signal
from scipy.stats import norm
import torch


class Comms_System:

    def __init__(self, symbol_set, symbol_seq, num_samples=8, beta=0.35, norm_h=True):
        self.symbol_set = symbol_set
        self.symbol_seq = symbol_seq
        self.symbol_seq_tensor = torch.Tensor(symbol_seq)
        self.m = num_samples
        self.h = self.rrcos(beta=beta)
        if norm_h:
            self.h = self.h / np.sqrt(np.sum(np.square(self.h)))  # normalize the filter
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


    def transmission(self, mode='euclidean', noise_level=2, norm_signal=False, model=None, v=True):

        # calculate gain_factor
        gain_factor = np.max(np.convolve(self.h, self.h))
        energy = np.mean(np.array(self.symbol_seq) ** 2)
        if v:
            print("E:", gain_factor)
            print('Ratio:', energy * gain_factor/noise_level**2)
        # upsample
        upsampled = self.upsample()

        # filter with rrcos
        Tx = np.convolve(upsampled, self.h)

        # normalize
        if norm_signal:
            Tx = Tx / np.sqrt(np.mean(np.square(Tx))) #np.sqrt(np.mean(np.square(Tx)))

        # Transmit / add noise
        Tx = Tx + np.random.normal(0.0, noise_level, Tx.shape)  # add gaussian noise

        # filter with rrcos on receiver
        if norm_signal:
            Rx = np.convolve(Tx, self.h) * np.sqrt(np.mean(self.symbol_seq**2))
        else:
            Rx = np.convolve(Tx, self.h)

        # downsample
        if norm_signal:
            downsampled = self.downsample(Rx)
        else:
            downsampled = self.downsample(Rx)/gain_factor

        if mode == 'euclidean':
            received_symbols = self.decision_making(downsampled, False)
        elif mode == 'network':
            received_symbols = network_receiver(Tx, self.symbol_set, model=model)

        return received_symbols


    def evaluate(self, decisions):

        num_errors = np.sum(np.array(decisions) != np.array(self.symbol_seq))
        error_rate = num_errors / len(self.symbol_seq)
        # print('{}% error rate'.format(np.round(error_rate*100, 4)))

        return num_errors, error_rate


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


def butter_lowpass(cutoff_freq, sampling_rate, order=4):

    nyquist_freq = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normalized_cutoff, 'low')
    return b, a


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

def SNR_plot_vanilla(num_symbols=10000, norm_h=False, normalized_network=False, norm_signal=False, use_gain=True, model=None):
    symbol_set = [3, 1, -1, -3]  # all symbols that we use
    symbol_seq = np.random.choice(symbol_set, num_symbols, replace=True)
    m = 8
    CS = Comms_System(symbol_set=symbol_set, symbol_seq=symbol_seq, num_samples=m, beta=0.35, norm_h=norm_h)

    SNRdbs = np.linspace(0, 18, 50)

    sigmas = []
    euclid_error_rates = []
    network_error_rates = []
    avg_symbol_energy = np.mean(np.array(symbol_seq) ** 2)
    print('Avg symbol energy', avg_symbol_energy)
    gain_factor = np.max(np.convolve(CS.h, CS.h))
    print('gain', gain_factor)

    for SNRdb in SNRdbs:
        if normalized_network:
            sigma_euclid = CS.SNRdb_to_sigma(SNRdb, avg_symbol_energy, use_gain=True) # symbol energy og gain
            sigma_network = CS.SNRdb_to_sigma(SNRdb, 8, use_gain=False) # fordi vi har normaliseret er sample energi sat til 1, og vi har 8 samples pr symbol, er avg_symbol_energy så 1*8
            euclid_decisions = CS.transmission(noise_level=sigma_euclid, norm_signal=False, v=False)
            network_decisions = CS.transmission(mode='network', noise_level=sigma_network, norm_signal=True, v=False, model=model)
            sigmas.append(sigma_euclid)
        else:
            sigma = CS.SNRdb_to_sigma(SNRdb, avg_symbol_energy, use_gain=True)
            euclid_decisions = CS.transmission(noise_level=sigma, norm_signal=False, v=False)
            network_decisions = CS.transmission(mode='network', noise_level=sigma, norm_signal=False, v=False, model=model)
            sigmas.append(sigma)

        euclid_error_rates.append(CS.evaluate(euclid_decisions)[1])
        network_error_rates.append(CS.evaluate(network_decisions)[1])

    sigmas = np.array(sigmas)
    error_theory = 1.5 * (1 - norm.cdf(np.sqrt(gain_factor / sigmas ** 2)))
    euclid_error_rates = np.array(euclid_error_rates)
    network_error_rates = np.array(network_error_rates)

    return SNRdbs, euclid_error_rates, network_error_rates, error_theory

    plt.figure(figsize=(18, 11))
    plt.title('Noise Plot', fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=20)
    plt.ylabel('$P_e$', fontsize=20)
    plt.semilogy(SNRdbs, euclid_error_rates)
    plt.semilogy(SNRdbs, error_theory)
    plt.semilogy(SNRdbs, network_error_rates)

    legend = ['Euclid', 'Theory']
    plt.legend(legend, fontsize=16)
    plt.show()







