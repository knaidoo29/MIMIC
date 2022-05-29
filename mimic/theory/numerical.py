import numpy as np
from scipy.interpolate import interp1d
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


def get_num_Dzk(redshift, kh, pks):
    """Numerically computes the scale-dependent growth functions D(z,k).

    Parameters
    ----------
    redshift : array
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
    cond = np.where(redshift == 0.)[0]
    pk_z_0 = pks[cond]
    Dzk = []
    for i in range(0, len(pks)):
        _D = np.sqrt(pks[i]/pk_z_0)
        Dzk.append(_D)
    Dzk = np.array(Dzk)
    return Dzk


def get_num_fz(redshift, Dz, **kwargs):
    """Numerically computes the linear growth rate.

    Parameters
    ----------
    redshift : array
        Redshift values.
    Dz : array
        Linear growth function.

    Returns
    -------
    fz : array
        Numerical growth rate f(z).
    """
    dz = z[1:] - z[:-1]
    if any(dz > 0.):
        usereverse = True
        z = np.copy(z[::-1])
        Dz = np.copy(Dz[::-1])
    a = basics.z2a(z)
    loga = np.log(a)
    logD = np.log(Dz)
    fz = num_diff(loga, logD, **kwargs)
    if usereverse:
        fz = fz[::-1]
    return fz


def get_num_fzk(redshift, kh, Dzk):
    """Numerically computed scale-dependent growth rate f(z,k).

    Parameters
    ----------
    redshift : array
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


def get_mean_Dz(redshift, kh, Dzk, kmin=0.001, kmax=10.):
    """Returns the mean growth function from the scale dependent growth
    function.

    Parameters
    ----------
    redshift : array
        Redshift values.
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
    return redshift, Dz


def get_mean_fz(redshift, kh, fzk, kmin=0.001, kmax=10.):
    """Returns the mean growth rate from the scale dependent growth rate.

    Parameters
    ----------
    redshift : array
        Redshift values.
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
    return redshift, fz
