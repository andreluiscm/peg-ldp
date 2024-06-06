import numpy as np


def geometric_noise(sensitivity:np.float_, epsilon:np.float_, n_samples=None) -> np.int_:
    p = 1 - np.exp(-epsilon / sensitivity)

    if n_samples is None:
        return np.random.geometric(p) - np.random.geometric(p)
    else:
        return np.random.geometric(p, n_samples) - np.random.geometric(p, n_samples)
