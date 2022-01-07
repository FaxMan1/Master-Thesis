from scipy import signal
import scipy

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


