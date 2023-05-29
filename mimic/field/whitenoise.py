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


def color_white_noise(wn, dx, kmag, interp_pk, mode='3D'):
    """Colours the Fourier modes of a white noise field to embue the desired
    power spectrum.

    Parameters
    ----------
    wn : ndarray
        Fourier modes of a white noise field.
    dx : float
        Size of the 3D grid.
    kmag : ndarray
        k-vector magnitude.
    interp_pk : function
        Power spectrum interpolation function.
    """
    dk = np.zeros(np.shape(wn)) + 1j*np.zeros(np.shape(wn))
    dk = np.sqrt(interp_pk(kmag)) * wn
    cond = np.where(kmag == 0.)
    dk[cond] = 0.
    if mode == '3D':
        dk /= np.sqrt(dx**3.)
    elif mode == '2D':
        dk /= np.sqrt(dx**2.)
    return dk
