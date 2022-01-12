from scipy import signal
import scipy
import torch

def butter_lowpass(cutoff_freq, sampling_rate, order=4):

    nyquist_freq = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normalized_cutoff, 'low')
    return b, a

def ideal_lowpass(signal, cutoff_freq, sampling_rate):
    Tx_freq = scipy.fft.rfft(signal)
    xf = scipy.fft.rfftfreq(signal.shape[0], 1 / sampling_rate)
    Tx_freq[xf > cutoff_freq] = 0
    Tx_low = scipy.fft.irfft(Tx_freq, n=signal.shape[0])
    return Tx_low

def ideal_lowpass_torch(signal, cutoff_freq, sample_rate):
    signal = signal.view(1,1,-1)
    Tx_freq = torch.fft.rfft(signal)
    xf = torch.fft.rfftfreq(signal.shape[2], 1 / sample_rate)
    Tx_freq[0][0][xf > cutoff_freq] = 0
    Tx_low = torch.fft.irfft(Tx_freq, n=signal.shape[2])
    return Tx_low



