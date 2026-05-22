import numpy as np
from numba import njit


@njit
def distance_1d_float(rx, boxsize):
    """Wrap a 1D separation into the periodic box.

    Parameters
    ----------
    rx : float
        Separation along one axis.
    boxsize : float
        Size of the periodic box.

    Returns
    -------
    float
        Wrapped separation in the range [-boxsize/2, boxsize/2].
    """
    if rx < -boxsize / 2.0:
        return rx + boxsize
    elif rx > boxsize / 2.0:
        return rx - boxsize
    else:
        return rx


@njit
def distance_3d_float(rx, ry, rz, boxsize):
    """Compute a periodic 3D separation and its magnitude.

    Parameters
    ----------
    rx, ry, rz : float
        Coordinate separations along each axis.
    boxsize : float
        Size of the periodic box.

    Returns
    -------
    tuple
        (r, newrx, newry, newrz) where r is the periodic distance and
        newrx/newry/newrz are the wrapped axis separations.
    """
    newrx = distance_1d_float(rx, boxsize)
    newry = distance_1d_float(ry, boxsize)
    newrz = distance_1d_float(rz, boxsize)
    newr = np.sqrt(newrx * newrx + newry * newry + newrz * newrz)
    return newr, newrx, newry, newrz


@njit
def get_vec_norm_float(x, y, z):
    """Normalize a 3D vector, with a fallback for zero length.

    Parameters
    ----------
    x, y, z : float
        Vector components.

    Returns
    -------
    tuple
        Normalized vector components (nx, ny, nz).
    """
    r = np.sqrt(x * x + y * y + z * z)
    if r == 0.0:
        c = 1.0 / np.sqrt(3.0)
        return c, c, c
    return x / r, y / r, z / r
