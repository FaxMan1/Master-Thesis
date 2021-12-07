from scipy import signal

def butter_lowpass(cutoff_freq, sampling_rate, order=4):

    nyquist_freq = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normalized_cutoff, 'low')
    return b, a
