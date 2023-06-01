import numpy as np

from .. import theory


def prep4MIMIC(redshift, kh, pks, Hz_func, fname_prefix, scaledep=False,
    kmin=0.001, kmax=10., return_outputs=False, **Hz_kwargs):
    """Function for returning expansion, growth rate from input power spectra and
    Hubble expansion rate function.

    Parameters
    ----------
    redshifts : array
        Redshift values.
    kh : array
        K values for the power spectrum.
    pks : array
        Power spectra arrays.
    Hz_func : function
        Hubble expansion rate function.
    fname_prefix : str
        Filename of outputs.
    scaledep : bool, optional
        Defines whether growth functions are scale dependent or not.
    kmin, kmax : float, optional
        Ranges used to calculate mean growth functions.
    return_outputs, bool, optional
        Determines whether to output calculated growth and expansion rates.
    **Hz_kwargs : kwargs, optional
        Keyword arguments for the Hz_func function.

    Returns
    -------
    pk0 : array
        Power spectra at redshift 0.
    Hz : array
        Hubble expansion rate.
    kh : array
        If scaldep is True, the fourier scales for the power spectra and
        growth functions
    Dzk : 2darray
        If scaldep is True, the scale dependent linear growth function.
    fzk : 2darray
        If scaldep is True, the scale dependent linear growth rate.
    Dz_mean : array
        If scaldep is False, the scale independent linear growth function.
    fz_mean : array
        If scaldep is False, the scale independent linear growth rate.
    """
    Hz = Hz_func(redshift, **Hz_kwargs)
    Dzk = theory.get_num_Dzk(redshift, kh, pks)
    fzk = theory.get_num_fzk(redshift, kh, Dzk)
    cond = np.where(redshift == 0.)[0]
    pk0 = np.copy(pks[cond].flatten())
    fname = fname_prefix + 'pk0.npz'
    np.savez(fname, kh=kh, pk=pk0)
    if scaledep:
        fname = fname_prefix + 'growth_scale_dep.npz'
        np.savez(fname, z=redshift, Hz=Hz, kh=kh, Dzk=Dzk, fzk=fzk)
        if return_outputs:
            return pk0, Hz, kh, Dzk, fzk
    else:
        Dz_mean = theory.get_mean_Dz(kh, Dzk, kmin=kmin, kmax=kmax)
        fz_mean = theory.get_mean_fz(kh, fzk, kmin=kmin, kmax=kmax)
        fname = fname_prefix + 'growth_scale_indep.npz'
        np.savez(fname, z=redshift, Hz=Hz, Dz=Dz_mean, fz=fz_mean)
        if return_outputs:
            return pk0, Hz, Dz_mean, fz_mean


def prep4MIMIC_LCDM(redshift, kh, pks, H0, Omega_cdm0, fname_prefix, scaledep=False,
    kmin=0.001, kmax=10., return_outputs=False, **Hz_kwargs):
    """Function for returning expansion, growth rate from input power spectra and
    with a LCDM Hubble expansion rate function.

    Parameters
    ----------
    redshifts : array
        Redshift values.
    kh : array
        K values for the power spectrum.
    pks : array
        Power spectra arrays.
    H0 : float
        Hubble expansion rate.
    Omega_cdm0 : float
        Matter density
    scaledep : bool, optional
        Defines whether growth functions are scale dependent or not.
    kmin, kmax : float, optional
        Ranges used to calculate mean growth functions.
    return_outputs, bool, optional
        Determines whether to output calculated growth and expansion rates.
    **Hz_kwargs : kwargs, optional
        Keyword arguments for the Hz_func function.

    Returns
    -------
    pk0 : array
        Power spectra at redshift 0.
    Hz : array
        Hubble expansion rate.
    kh : array
        If scaldep is True, the fourier scales for the power spectra and
        growth functions
    Dzk : 2darray
        If scaldep is True, the scale dependent linear growth function.
    fzk : 2darray
        If scaldep is True, the scale dependent linear growth rate.
    Dz_mean : array
        If scaldep is False, the scale independent linear growth function.
    fz_mean : array
        If scaldep is False, the scale independent linear growth rate.
    """
    Hz = theory.get_Hz_LCDM(redshift, H0, Omega_cdm0, **Hz_kwargs)
    Dzk = theory.get_num_Dzk(redshift, kh, pks)
    fzk = theory.get_num_fzk(redshift, kh, Dzk)
    cond = np.where(redshift == 0.)[0]
    pk0 = np.copy(pks[cond].flatten())
    fname = fname_prefix + 'pk0.npz'
    np.savez(fname, kh=kh, pk=pk0)
    if scaledep:
        fname = fname_prefix + 'growth_scale_dep.npz'
        np.savez(fname, z=redshift, Hz=Hz, kh=kh, Dzk=Dzk, fzk=fzk)
        if return_outputs:
            return pk0, Hz, kh, Dzk, fzk
    else:
        Dz_mean = theory.get_mean_Dz(kh, Dzk, kmin=kmin, kmax=kmax)
        fz_mean = theory.get_mean_fz(kh, fzk, kmin=kmin, kmax=kmax)
        fname = fname_prefix + 'growth_scale_indep.npz'
        np.savez(fname, z=redshift, Hz=Hz, Dz=Dz_mean, fz=fz_mean)
        if return_outputs:
            return pk0, Hz, Dz_mean, fz_mean
