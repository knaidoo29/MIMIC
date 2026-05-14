import numpy as np
from scipy.special import expit

from ...ext import shift


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
    #Below is the scipy implementation of 1./(np.exp((k-k0)/(k0*T)) + 1.), this is
    #more numerically stable and does not give an error for large values.
    return expit(-(k-k0)/(k0*T))


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
    return np.sqrt(1. - get_lowres_filter(k, lowres_k_nyq, k0=k0, T=T)**2.)


def upsample(dlowres, dhighres, ngridlow, ngridhigh, boxsize, MPI=None):
    if MPI is None:
        assert ngridhigh % ngridlow == 0, "Low resolution must be a factor of upsample resolution."
        x3Dlow, y3Dlow, z3Dlow = shift.cart.grid3D(boxsize, ngridlow)
        x3Dhigh, y3Dhigh, z3Dhigh = shift.cart.grid3D(boxsize, ngridhigh)
        dxlow = boxsize/ngridlow
        pixX = np.floor(x3Dhigh/dxlow).astype('int')
        pixY = np.floor(y3Dhigh/dxlow).astype('int')
        pixZ = np.floor(z3Dhigh/dxlow).astype('int')
        dlow2highres = dlowres[pixX, pixY, pixZ]
        dlow2highresk = shift.cart.fft3D(dlow2high, boxsize)
        dhighresk = shift.cart.fft3D(dhighres, boxsize)
        kx3D, ky3D, kz3D = shift.cart.kgrid3D(boxsize, ngrid)
        kmag = np.sqrt(kx3D, ky3D, kz3D)
        lowres_k_nyq = shift.cart.get_kn(boxsize, ngrid)
        dlowupsamplek = dlow2highresk*get_lowres_filter(kmag, lowres_k_nyq) + dhighresk*get_highres_filter(kmag, lowres_k_nyq)
        dlowupsample = shift.cart.ifft3D(dlowupsamplek, boxsize)
    return dlowupsample


    # need to distribute highres points to lowres.
    # sample points
    # redistribute to highres mpi
