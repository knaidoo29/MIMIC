import numpy as np
from numba import njit

from .analytic_correlator import get_cc_array2

from .progress import progress_bar


@njit
def corr_dot_eta(
    x1, xc, y1, yc, z1, zc, ex1, exc, ey1, eyc, ez1, ezc,
    type1, typec, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u,
    psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu,
    boxsize, eta, mpi_rank=0, lenpro=0, prefix=''
):
    """Compute the projected eta-weighted field for a single point and constraint set.

    Parameters
    ----------
    x1, y1, z1 : array
        Coordinates of point 1 values.
    xc, yc, zc : array
        Coordinates of constraint points.
    ex1, ey1, ez1 : float
        Direction vector for point 1.
    exc, eyc, ezc : array
        Constraint direction vectors.
    type1 : int
        Type for point 1.
    typec : array
        Types for constraint points.
    adot_phi, adot_vel : float
        Scaling factors for displacements and velocities.
    logr, xi, zeta_p, zeta_u, psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu : arrays
        Correlation profiles.
    boxsize : float
        Periodic box size.
    lenr : int
        Length of correlation arrays.
    lenx1 : int
        Number of point 1 values.
    lenxc : int
        Number of constraints.
    eta : array
        Weight values for constraints.
    mpi_rank : int, optional
        MPI rank for progress output.
    lenpro : int, optional
        Progress bar width.
    prefix : str, optional
        Progress bar prefix.

    Returns
    -------
    field : array
        Field values.
    """
    field = np.empty(len(x1), dtype=np.float64)

    for i in range(0, len(x1)):
        cc = get_cc_array2(
            x1[i], xc, y1[i], yc, z1[i], zc, ex1, exc, ey1, eyc, ez1, ezc,
            type1, typec, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u,
            psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu, boxsize)
        field[i] = np.dot(cc, eta)
        if mpi_rank == 0:
            progress_bar(i + 1, len(x1), lenpro, prefix)

    return field


@njit
def corr_dot_eta_array(
    x1, xc, y1, yc, z1, zc, ex1, exc, ey1, eyc, ez1, ezc, type1, typec, adot_phi, adot_vel, 
    logr, xi, zeta_p, zeta_u, psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu,
    boxsize, eta, mpi_rank=0, lenpro=0, prefix=''
):
    """Compute the projected eta-weighted field when point 1 has array constraints.

    Parameters
    ----------
    x1, y1, z1 : array
        Coordinates of point 1 values.
    xc, yc, zc : array
        Coordinates of constraint points.
    ex1, ey1, ez1 : array
        Direction vectors for point 1 values.
    exc, eyc, ezc : array
        Constraint direction vectors.
    type1 : array
        Types for point 1 values.
    typec : array
        Types for constraint points.
    adot_phi, adot_vel : float
        Scaling factors for displacements and velocities.
    logr, xi, zeta_p, zeta_u, psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu : arrays
        Correlation profiles.
    boxsize : float
        Periodic box size.
    eta : array
        Weight values for constraints.
    mpi_rank : int, optional
        MPI rank for progress output.
    lenpro : int, optional
        Progress bar width.
    prefix : str, optional
        Progress bar prefix.

    Returns
    -------
    array
        Weighted field values.
    """
    field = np.empty(len(x1), dtype=np.float64)

    for i in range(0, len(x1)):
        cc = get_cc_array2(
            x1[i], xc, y1[i], yc, z1[i], zc, ex1[i], exc, ey1[i], eyc, ez1[i], ezc,
            type1, typec, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u,
            psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu, boxsize
        )
        field[i] = np.dot(cc, eta)
        if mpi_rank == 0:
            progress_bar(i + 1, len(x1), lenpro, prefix)

    return field
