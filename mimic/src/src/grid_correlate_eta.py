import numpy as np
from numba import njit

from .grid_correlator import get_cc_grid_array2

from .progress import progress_bar


@njit
def corr_dot_eta_grid(
    x1, xc, y1, yc, z1, zc, ex1, exc, ey1, eyc, ez1, ezc,
    type1, typec, adot_phi, adot_vel, xi_box, zeta_p_box, zeta_u_box, 
    psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box,
    ngrid, boxsize, eta, stretch, mpi_rank=0, lenpro=0, prefix=''
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
    xi_box, zeta_p_box, zeta_u_box, psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box : arrays
        Correlation profiles on a grid.
    ngrid : int
        Number of grid points along each axis for the correlation grid.
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
    field : array
        Weighted field values.
    """
    field = np.empty(len(x1), dtype=np.float64)

    for i in range(0,len(x1)):
        cc = get_cc_grid_array2(
            x1[i], xc, y1[i], yc, z1[i], zc, ex1, exc, ey1, eyc, ez1, ezc,
            type1, typec, adot_phi, adot_vel, xi_box, zeta_p_box, zeta_u_box, 
            psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, 
            psixx_uu_box, psixy_uu_box, ngrid, boxsize, stretch
        )
        field[i] = np.dot(cc, eta)
        if mpi_rank == 0:
            progress_bar(i + 1, len(x1), lenpro, prefix)

    return field


@njit
def corr_dot_eta_array_grid(
    x1, xc, y1, yc, z1, zc, ex1, exc, ey1, eyc, ez1, ezc,
    type1, typec, adot_phi, adot_vel, xi_box, zeta_p_box, zeta_u_box, 
    psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box,
    ngrid, boxsize, eta, stretch, mpi_rank=0, lenpro=0, prefix=''
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
    xi_box, zeta_p_box, zeta_u_box, psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box : arrays
        Correlation profiles on a grid.
    ngrid : int
        Number of grid points along each axis for the correlation grid.
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
    field : array
        Weighted field values.
    """
    field = np.empty(len(x1), dtype=np.float64)

    for i in range(len(x1)):
        cc = get_cc_grid_array2(
            x1[i], xc, y1[i], yc, z1[i], zc, ex1[i], exc, ey1[i], eyc, ez1[i], ezc,
            type1, typec, adot_phi, adot_vel, xi_box, zeta_p_box, zeta_u_box,
            psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box,
            psixx_uu_box, psixy_uu_box, ngrid, boxsize, stretch
        )
        field[i] = np.dot(cc, eta)
        if mpi_rank == 0:
            progress_bar(i + 1, len(x1), lenpro, prefix)

    return field
