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