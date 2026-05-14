import numpy as np

from ..io import isscalar


def distance_1D(x1, x2, boxsize=None):
    """Returns the distance between points along 1D axis. If a boxsize is supplied
    then periodic conditions are applied to the distances.

    Parameters
    ----------
    x1, x2 : float/array
        Coordinates of points 1 and 2 in 1D coordinates.
    boxsize : float
        Size of periodic conditions on the 1D axis.

    Returns
    -------
    rx : float/array
        Distance along between x1 and x2.
    """
    rx = x2 - x1
    if boxsize is not None:
        if isscalar(rx):
            if rx < -boxsize/2:
                rx += boxsize
            elif rx > boxsize/2:
                rx -= boxsize
        else:
            cond = np.where(rx < -boxsize/2)
            rx[cond] += boxsize
            cond = np.where(rx > boxsize/2)
            rx[cond] -= boxsize
    return rx


def distance_3D(x1, x2, y1, y2, z1, z2, boxsize=None, return_axis_dist=False):
    """Enforces periodic boundary conditions along 3D cartesian coordinates.

    Parameters
    ----------
    x1, x2, y1, y2, z1, z2 : float/array
        Coordinates of points 1 and 2 in 3D coordinates.
    boxsize : float
        Size of periodic conditions on the 1D axis.

    Returns
    -------
    r : float/array
        Distance between 1 and 2.
    """

    rx = distance_1D(x1, x2, boxsize=boxsize)
    ry = distance_1D(y1, y2, boxsize=boxsize)
    rz = distance_1D(z1, z2, boxsize=boxsize)
    r = np.sqrt(rx**2. + ry**2. + rz**2.)
    if return_axis_dist:
        return rx, ry, rz, r
    else:
        return r
