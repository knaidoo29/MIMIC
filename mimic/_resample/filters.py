import numpy as np


def get_lowres_filter(k, lowres_k_nyq, k0=None, T=0.1):
    """Returns the low-resolution Fourier filter.

    Parameters
    ----------
    k : array
        Fourier modes.
    lowres_k_nyq : float
        Nyquist frequency for low-resolution dataset.
    k0 : float, optional
        Low-resolution filter cut out.
    T : float, optional
        'Temperature'.
    """
    if k0 is None:
        k0 = 0.5*lowres_k_nyq
    return 1./(np.exp((k-k0)/(k0*T)) + 1.)


def get_highres_filter(k, lowres_k_nyq, k0=None, T=0.1):
    """Returns the high-resolution Fourier filter.

    Parameters
    ----------
    k : array
        Fourier modes.
    lowres_k_nyq : float
        Nyquist frequency for low-resolution dataset.
    k0 : float, optional
        Low-resolution filter cut out.
    T : float, optional
        'Temperature'.
    """
    if k0 is None:
        k0 = 0.5*lowres_k_nyq
    return np.sqrt(1. - get_lowk_filter(k, lowres_k_nyq, k0=k0, T=T)**2.)
