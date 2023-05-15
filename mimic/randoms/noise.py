import numpy as np


def get_white_noise(seed, *args):
    """Returns white noise, standard normal distribution with mean = 0 and
    variance = 1.

    Parameters
    ----------
    seed : int
        Numpy random seed.

    Returns
    -------
    wn : ndarray
        White noise in real coordinates.
    """
    np.random.seed(seed)
    wn = np.random.randn(*args)
    return wn
