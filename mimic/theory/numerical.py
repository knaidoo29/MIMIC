import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from . import basics


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
