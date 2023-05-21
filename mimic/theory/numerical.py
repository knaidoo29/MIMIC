import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import simps

from . import basics
from . import expansion


def num_diff(x, f, equal_spacing=False, interpgrid=1000, kind='cubic'):
    """For unequally spaced data we interpolate onto an equal spaced 1d grid
    which we then use the symmetric two-point derivative and the non-symmetric
    three point derivative estimator.

    Parameters
    ----------
    x : array
        X-axis.
    f : array
        Function values at x.
    equal_spacing : bool, optional
        Automatically assumes data is not equally spaced and will interpolate
        from it.
    interp1dgrid : int, optional
        Grid spacing for the interpolation grid, if equal spacing is False.
    kind : str, optional
        Interpolation kind.

    Returns
    -------
    df : array
        Numerical differentiation values for f evaluated at points x.

    Notes
    -----
    For non-boundary values:

    df   f(x + dx) - f(x - dx)
    -- = ---------------------
    dx            2dx

    For boundary values:

    df   - f(x + 2dx) + 4f(x + dx) - 3f(x)
    -- = ---------------------------------
    dx                  2dx
    """
    if equal_spacing == False:
        interpf = interp1d(x, f, kind=kind)
        x_equal = np.linspace(x.min(), x.max(), interpgrid)
        f_equal = interpf(x_equal)
    else:
        x_equal = np.copy(x)
        f_equal = np.copy(f)
    dx = x_equal[1] - x_equal[0]
    df_equal = np.zeros(len(x_equal))
    # boundary differentials
    df_equal[0] = (-f_equal[2] + 4*f_equal[1] - 3.*f_equal[0])/(2.*dx)
    df_equal[-1] = (f_equal[-3] - 4*f_equal[-2] + 3.*f_equal[-1])/(2.*dx)
    # non-boundary differentials
    df_equal[1:-1] = (f_equal[2:] - f_equal[:-2])/(2.*dx)
    if equal_spacing == False:
        interpdf = interp1d(x_equal, df_equal, kind=kind)
        df = interpdf(x)
    else:
        df = np.copy(df_equal)
    return df


def get_num_Dzk(redshifts, kh, pks):
    """Numerically computes the scale-dependent growth functions D(z,k).

    Parameters
    ----------
    redshifts : array
        Redshift values.
    kh : array
        K values for the power spectrum.
    pks : array
        Power spectra arrays.

    Returns
    -------
    Dzk : 2darray
        Numerical the scale-dependent growth functions.
    """
    cond = np.where(redshifts == 0.)[0][0]
    pk_z_0 = pks[cond]
    Dzk = []
    for i in range(0, len(pks)):
        _D = np.sqrt(pks[i]/pk_z_0)
        Dzk.append(_D)
    Dzk = np.array(Dzk)
    return Dzk


def get_num_fz(redshifts, Dz, **kwargs):
    """Numerically computes the linear growth rate.

    Parameters
    ----------
    redshifts : array
        Redshift values.
    Dz : array
        Linear growth function.

    Returns
    -------
    fz : array
        Numerical growth rate f(z).
    """
    dz = redshifts[1:] - redshifts[:-1]
    if any(dz > 0.):
        usereverse = True
        redshifts = np.copy(redshifts[::-1])
        Dz = np.copy(Dz[::-1])
    else:
        usereverse = False
    a = basics.z2a(redshifts)
    loga = np.log(a)
    logD = np.log(Dz)
    fz = num_diff(loga, logD, **kwargs)
    if usereverse:
        fz = fz[::-1]
    return fz


def get_num_fzk(redshifts, kh, Dzk):
    """Numerically computed scale-dependent growth rate f(z,k).

    Parameters
    ----------
    redshifts : array
        Redshift values.
    kh : array
        K values for the power spectrum.
    pks : array
        Power spectra arrays.

    Returns
    -------
    fzk : 2darray
        Numerical the scale-dependent growth functions.
    """
    fzk = np.zeros(np.shape(Dzk))
    for i in range(0, len(Dzk[0])):
        fzk[:, i] = get_num_fz(redshifts, Dzk[:, i])
    return fzk


def get_mean_Dz(kh, Dzk, kmin=0.001, kmax=10.):
    """Returns the mean growth function from the scale dependent growth
    function.

    Parameters
    ----------
    kh : array
        K values for the power spectrum.
    Dzk : 2darray
        Numerical the scale-dependent growth functions.
    kmin, kmax : float, optional
        k-range for computing the mean.

    Returns
    -------
    Dz : array
        Numerical mean growth function D(z).
    """
    cond = np.where((kh >= kmin) & (kh <= kmax))[0]
    Dz = np.mean(Dzk[:,cond], axis=1)
    return Dz


def get_mean_fz(kh, fzk, kmin=0.001, kmax=10.):
    """Returns the mean growth rate from the scale dependent growth rate.

    Parameters
    ----------
    kh : array
        K values for the power spectrum.
    fzk : 2darray
        Numerical the scale-dependent growth rate.
    kmin, kmax : float, optional
        k-range for computing the mean.

    Returns
    -------
    fz : array
        Numerical mean growth rate f(z).
    """
    cond = np.where((kh >= kmin) & (kh <= kmax))[0]
    fz = np.mean(fzk[:,cond], axis=1)
    return fz


def get_sigmaR(kh, pk, R=8., kmin=None, kmax=None):
    """Returns sigmaR, default is sigma8.

    Parameters
    ----------
    kh : array
        K values for the power spectrum.
    pk : array
        Power spectrum.
    R : float, optional
        Scale, default is 8.
    kmin : float, optional
        Minimum k for sigmaR integration.
    kmax : float, optional
        Maximum k for sigmaR integration.

    Returns
    -------
    sigmaR : float
        The value of sigmaR.

    Notes
    -----
    Calculating sigmaR:
    sigma_R = [int_0 ^infty w^{2}(kR) Deltak^{2} dk]^0.5
    Where:
    w(x) = (3/x^3)*(sin(x) - x*cos(x))
    Deltak^2 = k^{2} P(k)/(2pi^2)
    """
    # First we're going to interpolate values of the power spectrum with equal
    # spaced bins.
    interp_PK = interp1d(kh, pk, kind='cubic')
    if kmin is None:
        kmin = kh.min()
    if kmax is None:
        kmax = kh.max()
    kscale = 2.*np.pi/R
    kbins = 100*int((kmax - kmin)/kscale)
    k = np.linspace(kmin, kmax, kbins)
    p = interp_PK(k)
    Deltak2 = (k**2.)*p/(2.*(np.pi**2.))
    wkR = (3./((k*R)**3.))*(np.sin(k*R) - (k*R)*np.cos(k*R))
    # Use scipy sample integration evaluation using the simpson rule
    sigmaR2 = simps((wkR**2.)*Deltak2, k)
    sigmaR = np.sqrt(sigmaR2)
    return sigmaR


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
    Dzk = get_num_Dzk(redshift, kh, pks)
    fzk = get_num_fzk(redshift, kh, Dzk)
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
        Dz_mean = get_mean_Dz(kh, Dzk, kmin=kmin, kmax=kmax)
        fz_mean = get_mean_fz(kh, fzk, kmin=kmin, kmax=kmax)
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
    Hz = expansion.get_Hz_LCDM(redshift, H0, Omega_cdm0, **Hz_kwargs)
    Dzk = get_num_Dzk(redshift, kh, pks)
    fzk = get_num_fzk(redshift, kh, Dzk)
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
        Dz_mean = get_mean_Dz(kh, Dzk, kmin=kmin, kmax=kmax)
        fz_mean = get_mean_fz(kh, fzk, kmin=kmin, kmax=kmax)
        fname = fname_prefix + 'growth_scale_indep.npz'
        np.savez(fname, z=redshift, Hz=Hz, Dz=Dz_mean, fz=fz_mean)
        if return_outputs:
            return pk0, Hz, Dz_mean, fz_mean
