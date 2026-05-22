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


def get_white_noise_3D(seed, ngrid, MPI=None):
    """Returns unique and reproducible white noise field in 3D. Seed is used to
    generate a unique seed for each layer of the grid which is then in turn
    used to generate a random field.

    Parameters
    ----------
    seed : int
        Numpy random seed.
    ngrid : int
        Grid dimensions.
    MPI : object
        MPIutils mpi object class.

    Returns
    -------
    wn : ndarray
        White noise in real coordinates.
    """
    # Don't change this, otherwise seeds won't match!
    seedmax=1_000_000_000
    np.random.seed(seed)
    seeds = np.random.randint(seedmax, size=ngrid)
    if MPI is not None:
        seeds = MPI.split_array(seeds)
    wn = np.concatenate([get_white_noise(_seed, *(1, ngrid, ngrid)) for _seed in seeds])
    return wn


def color_white_noise(wnk, dx, kmag, interp_pk, mode='3D'):
    """Colours the Fourier modes of a white noise field to embue the desired
    power spectrum.

    Parameters
    ----------
    wnk : ndarray
        Fourier modes of a white noise field.
    dx : float
        Size of the 3D grid.
    kmag : ndarray
        k-vector magnitude.
    interp_pk : function
        Power spectrum interpolation function.

    Returns
    -------
    dk : ndarray
        Fourier modes of a Gaussian random field.
    """
    dk = np.zeros(np.shape(wnk)) + 1j*np.zeros(np.shape(wnk))
    dk = np.sqrt(interp_pk(kmag)) * wnk
    cond = np.where(kmag == 0.)
    dk[cond] = 0.
    if mode == '3D':
        dk /= np.sqrt(dx**3.)
    elif mode == '2D':
        dk /= np.sqrt(dx**2.)
    return dk


def whitten_colored_field(dk, dx, kmag, interp_pk, mode='3D'):
    """Whittens the Fourier modes of a gaussian random field to get the Fourier
    modes of the white noise field.

    Parameters
    ----------
    dk : ndarray
        Fourier modes of a Gaussian random field.
    dx : float
        Size of the grid.
    kmag : ndarray
        k-vector magnitude.
    interp_pk : function
        Power spectrum interpolation function.

    Returns
    -------
    wnk : ndarray
        Fourier modes of a white noise field.
    """
    wnk = np.zeros(np.shape(dk)) + 1j*np.zeros(np.shape(dk))
    cond = np.where(kmag != 0.)
    wnk[cond] = dk[cond]/np.sqrt(interp_pk(kmag[cond]))
    if mode == '3D':
        wnk *= np.sqrt(dx**3.)
    elif mode == '2D':
        wnk *= np.sqrt(dx**2.)
    return wnk
