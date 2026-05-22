import numpy as np
from numba import njit


@njit
def interp_lin_float(x, f, xval, fillval):
    """Linearly interpolate a scalar value on a uniformly spaced grid.

    Parameters
    ----------
    x : array
        Grid coordinates.
    f : array
        Function values at grid points.
    xval : float
        Query coordinate.
    fillval : float
        Value returned outside of the grid range.

    Returns
    -------
    float
        Interpolated value.
    """
    lenx = x.shape[0]
    xmin = x[0]
    xmax = x[lenx - 1]
    dx = (xmax - xmin) / (lenx - 1)

    if xval < xmin:
        return fillval
    if xval > xmax:
        return fillval

    ind = int(np.floor((xval - xmin) / dx))
    if ind >= lenx - 1:
        return f[lenx - 1]

    x1 = x[ind]
    x2 = x[ind + 1]
    f1 = f[ind]
    f2 = f[ind + 1]
    return f1 + (f2 - f1) * (xval - x1) / (x2 - x1)


@njit
def interp_lin_array(x, f, xarr, fillval):
    """Interpolate a vector of query points on a uniform grid.

    Parameters
    ----------
    x : array
        Grid coordinates.
    f : array
        Function values.
    xarr : array
        Query coordinates.
    fillval : float
        Value returned outside of the grid range.

    Returns
    -------
    array
        Interpolated values.
    """
    lenxarr = xarr.shape[0]
    farr = np.empty(lenxarr, dtype=f.dtype)
    for i in range(lenxarr):
        farr[i] = interp_lin_float(x, f, xarr[i], fillval)
    return farr


@njit
def interp_log_float(logx, f, logxval, fmin, fmax):
    """Interpolate a scalar value on a logarithmically spaced grid.

    Parameters
    ----------
    logx : array
        Logarithmic grid coordinates.
    f : array
        Function values at grid points.
    logxval : float
        Query log coordinate.
    fmin : float
        Fill value below the grid minimum.
    fmax : float
        Fill value above the grid maximum.

    Returns
    -------
    float
        Interpolated value.
    """
    lenx = logx.shape[0]
    logxmin = logx[0]
    logxmax = logx[lenx - 1]
    dlogx = (logxmax - logxmin) / (lenx - 1)

    if logxval <= logxmin:
        return fmin
    if logxval >= logxmax:
        return fmax

    ind = int(np.floor((logxval - logxmin) / dlogx))
    if ind >= lenx - 1:
        return fmax

    logx1 = logx[ind]
    logx2 = logx[ind + 1]
    f1 = f[ind]
    f2 = f[ind + 1]
    return f1 + (f2 - f1) * (logxval - logx1) / (logx2 - logx1)


@njit
def interp_log_array(logx, f, logxarr, fmin, fmax):
    """Interpolate an array of log-coordinates on a log-spaced grid.

    Parameters
    ----------
    logx : array
        Logarithmic grid coordinates.
    f : array
        Function values at grid nodes.
    logxarr : array
        Query log coordinates.
    fmin : float
        Value returned below the grid minimum.
    fmax : float
        Value returned above the grid maximum.

    Returns
    -------
    array
        Interpolated values.
    """
    lenxarr = logxarr.shape[0]
    farr = np.empty(lenxarr, dtype=f.dtype)
    for i in range(lenxarr):
        farr[i] = interp_log_float(logx, f, logxarr[i], fmin, fmax)
    return farr
