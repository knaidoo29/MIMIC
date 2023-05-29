import numpy as np

from scipy.interpolate import interp1d


def Dzk_2_Dk_at_z(zval, zarray, karray, Dzk):
    """Returns an interpolated function for the scale dependent linear growth
    function at redshift zval.

    Parameters
    ----------
    zval : float
        Redshift you want the scale dependent linear growth function.
    zarray : array
        Redshift values for the scale dependent linear growth function.
    karray : array
        Fourier modes for the scale dependent linear growth function.
    Dzk : array
        Scale dependent growth function evaluated across several redshifts.

    Returns
    -------
    interp_Dk : func
        Interpolatable function for the scale dependent linear growth function.
    """
    Dk = np.zeros(len(kh))
    for i in range(0, len(kh)):
        interp_D = interp1d(z, Dzk[:, i], kind='cubic')
        Dk[i] = interp_D(redshift)
    interp_Dk = interp1d(kh, Dk, kind='cubic', bounds_error=False)
    return interp_Dk


def Dz_2_interp_Dz(zarray, Dz):
    """Returns an interpolated function for the scale independent linear growth
    function as a function of redshift.

    Parameters
    ----------
    zarray : array
        Redshift values for the scale independent linear growth function.
    Dz : array
        Scale independent growth function.

    Returns
    -------
    interp_Dz : func
        Interpolatable function for the scale independent linear growth function.
    """
    interp_Dz = interp1d(zarray, Dz, kind='cubic')
    return interp_Dz



def get_growth_D(zval, zarray, Dzk, kval=None, karray=None):
    """Returns the linear growth function from tabulated scale dependent and
    independent linear growth functions.

    Parameters
    ----------
    zval : float
        Redshift you want the scale dependent linear growth function.
    zarray : array
        Redshift values for the scale dependent linear growth function.
    Dzk : array
        The scale dependent or independent growth function evaluated across several
        redshifts.
    kval : array, optiona
        Fourier modes you want the scale dependent linear growth function.
    karray : array, optiona
        Fourier modes for the scale dependent linear growth function, if Dzk is
        scale dependent.
    """
    if len(np.shape(Dzk)) == 2:
        error_string = " Dzk shape implies this scale dependence but no Fourier modes are provided."
        assert kval is not None and karray is not None, error_string
        interp_Dk = Dzk_2_Dk_at_z(zval, zarray, karray, Dzk)
        return interp_Dk(kval)
    else:
        interp_Dz = Dz_2_interp_Dz(zarray, Dzk)
        return interp_Dz(zval)


def fzk_2_fk_at_z(zval, zarray, karray, fzk):
    """Returns an interpolated function for the scale dependent linear growth
    rate at redshift zval.

    Parameters
    ----------
    zval : float
        Redshift you want the scale dependent linear growth rate.
    zarray : array
        Redshift values for the scale dependent linear growth rate.
    karray : array
        Fourier modes for the scale dependent linear growth rate.
    fzk : array
        Scale dependent growth rate evaluated across several redshifts.

    Returns
    -------
    interp_fk : func
        Interpolatable function for the scale dependent linear growth rate.
    """
    fk = np.zeros(len(kh))
    for i in range(0, len(kh)):
        interp_f = interp1d(zarray, fzk[:, i], kind='cubic')
        fk[i] = interp_f(zval)
    interp_fk = interp1d(kh, fk, kind='cubic', bounds_error=False)
    return interp_fk


def fz_2_interp_fz(zarray, fz):
    """Returns an interpolated function for the scale independent linear growth
    rate as a function of redshift.

    Parameters
    ----------
    zarray : array
        Redshift values for the scale independent linear growth function.
    fz : array
        Scale independent growth rate.

    Returns
    -------
    interp_fz : func
        Interpolatable function for the scale independent linear growth rate.
    """
    interp_fz = interp1d(zarray, fz, kind='cubic')
    return interp_fz


def get_growth_f(zval, zarray, fzk, kval=None, karray=None):
    """Returns the linear growth rate from tabulated scale dependent and
    independent linear growth rate.

    Parameters
    ----------
    zval : float
        Redshift you want the scale dependent linear growth rate.
    zarray : array
        Redshift values for the scale dependent linear growth rate.
    fzk : array
        The scale dependent or independent growth rate evaluated across several
        redshifts.
    kval : array, optiona
        Fourier modes you want the scale dependent linear growth function.
    karray : array, optiona
        Fourier modes for the scale dependent linear growth function, if Dzk is
        scale dependent.
    """
    if len(np.shape(fzk)) == 2:
        error_string = " fzk shape implies this scale dependence but no Fourier modes are provided."
        assert kval is not None and karray is not None, error_string
        interp_fk = fzk_2_fk_at_z(zval, zarray, karray, fzk)
        return interp_fk(kval)
    else:
        interp_fz = fz_2_interp_fz(zarray, fzk)
        return interp_fz(zval)
