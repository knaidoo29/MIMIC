import numpy as np
from scipy.integrate import simpson

from ...ext import shift


def get_sigma_8(kh, pk):
    """Calculates sigma_8 from an input power spectrum.

    Parameters
    ----------
    kh, pk : array_like
        The linear power spectrum and corresponding k value.

    Returns
    -------
    sigma_8 : float
        The calculated value of sigma_8.
    """
    w = 3.*(np.sin(kh*8.)-(kh*8.*np.cos(kh*8.)))/(kh*8.)**3.
    integrand = kh*kh*pk*w*w
    sigma_8_2 = (1./(2.*np.pi**2.))*simpson(integrand, kh)
    sigma_8 = sigma_8_2**0.5
    return sigma_8


def correct_pk_4_sigma8(kh, pk, boxsize, ngrid, sigma8=None):
    """Corrects power spectram for limits in boxsize.

    Parameters
    ----------
    kh, pk : array_like
        The linear power spectrum and corresponding k value.
    boxsize : float
        Size of the simulation box.
    ngrid : float
        Resolution of the simulation.
    sigma8 : float, optional
        Pre-specified sigma8 value to correct to.

    Returns
    -------
    pk_corr : array_like
        Corrected power spectrum.
    """
    pk_init = np.copy(pk)
    if sigma8 is None:
        sigma8 = get_sigma_8(kh, pk_init)
    pk_init_star = np.zeros(len(pk_init))
    kf = shift.cart.get_kf(boxsize)
    kn = shift.cart.get_kn(boxsize, ngrid)
    cond = np.where((kh >= kf) & (kh < kn))[0]
    pk_init_star[cond] = pk_init[cond]
    sigma8_star = get_sigma_8(kh, pk_init_star)
    pk_corr = pk_init * (sigma8/sigma8_star)**2.
    return pk_corr
