import numpy as np

from .fast_correlate import get_cc_array1, get_cc_array2
from .progress import progress_bar


def corr1_dot_inv_dot_corr2(x1, x2, y1, y2, z1, z2,
                             exi, eyi, ezi,
                             xc, yc, zc, exc, eyc, ezc,
                             type1, type2, typec,
                             adot_phi, adot_vel, logr, xi,
                             zeta_p, zeta_u,
                             psir_pp, psit_pp, psir_pu, psit_pu,
                             psir_uu, psit_uu,
                             boxsize, lenr, lenxi, lenxc,
                             inv, field=None,
                             mpi_rank=0, lenpro=0, lenpre=0, prefix=''):
    """Compute the correlation of point 1 and point 2 through inverse covariance.

    Parameters
    ----------
    x1, y1, z1 : array
        Coordinates of points set 1.
    x2, y2, z2 : array
        Coordinates of points set 2.
    exi, eyi, ezi : float
        Direction vector for point 1 constraints.
    exc, eyc, ezc : array
        Constraint direction vectors.
    type1 : int
        Type for point 1 values.
    type2 : int
        Type for point 2 values.
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
    lenxi : int
        Number of evaluation points.
    lenxc : int
        Number of constraints.
    inv : array
        Flattened inverse covariance matrix.
    field : array, optional
        Output array to fill.
    mpi_rank : int, optional
        MPI rank for progress output.
    lenpro : int, optional
        Progress bar width.
    lenpre : int, optional
        Length of prefix string.
    prefix : str, optional
        Progress bar prefix.

    Returns
    -------
    array
        Resulting correlated field values.
    """
    if field is None:
        field = np.empty(lenxi, dtype=np.float64)

    inv_mat = inv.reshape((lenxc, lenxc))

    for i in range(lenxi):
        cc1 = get_cc_array2(x1[i], xc, y1[i], yc, z1[i], zc,
                            exi, exc, eyi, eyc, ezi, ezc,
                            type1, typec, adot_phi, adot_vel,
                            logr, xi, zeta_p, zeta_u,
                            psir_pp, psit_pp, psir_pu, psit_pu,
                            psir_uu, psit_uu, boxsize)
        cc2 = get_cc_array1(xc, x2[i], yc, y2[i], zc, z2[i],
                            exc, exi, eyc, eyi, ezc, ezi,
                            typec, type2, adot_phi, adot_vel,
                            logr, xi, zeta_p, zeta_u,
                            psir_pp, psit_pp, psir_pu, psit_pu,
                            psir_uu, psit_uu, boxsize)
        field[i] = np.dot(cc1, inv_mat.dot(cc2))
        if mpi_rank == 0:
            progress_bar(i + 1, lenxi, lenpro, prefix)

    return field


def corr1_dot_inv_dot_corr2_array(x1, x2, y1, y2, z1, z2,
                                  exi, eyi, ezi,
                                  xc, yc, zc, exc, eyc, ezc,
                                  type1, type2, typec,
                                  adot_phi, adot_vel, logr, xi,
                                  zeta_p, zeta_u,
                                  psir_pp, psit_pp, psir_pu, psit_pu,
                                  psir_uu, psit_uu,
                                  boxsize, lenr, lenxi, lenxc,
                                  inv, field=None,
                                  mpi_rank=0, lenpro=0, lenpre=0, prefix=''):
    """Compute the inverse-covariance-correlated field when input vectors are arrays.

    Parameters
    ----------
    x1, y1, z1 : array
        Coordinates of points set 1.
    x2, y2, z2 : array
        Coordinates of points set 2.
    exi, eyi, ezi : array
        Direction vectors for point 1 values.
    exc, eyc, ezc : array
        Constraint direction vectors.
    type1 : array
        Types for point 1 values.
    type2 : array
        Types for point 2 values.
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
    lenxi : int
        Number of evaluation points.
    lenxc : int
        Number of constraints.
    inv : array
        Flattened inverse covariance matrix.
    field : array, optional
        Output array to fill.
    mpi_rank : int, optional
        MPI rank for progress output.
    lenpro : int, optional
        Progress bar width.
    lenpre : int, optional
        Length of prefix string.
    prefix : str, optional
        Progress bar prefix.

    Returns
    -------
    array
        Resulting correlated field values.
    """
    if field is None:
        field = np.empty(lenxi, dtype=np.float64)

    inv_mat = inv.reshape((lenxc, lenxc))

    for i in range(lenxi):
        cc1 = get_cc_array2(x1[i], xc, y1[i], yc, z1[i], zc,
                            exi[i], exc, eyi[i], eyc, ezi[i], ezc,
                            type1, typec, adot_phi, adot_vel,
                            logr, xi, zeta_p, zeta_u,
                            psir_pp, psit_pp, psir_pu, psit_pu,
                            psir_uu, psit_uu, boxsize)
        cc2 = get_cc_array1(xc, x2[i], yc, y2[i], zc, z2[i],
                            exc, exi[i], eyc, eyi[i], ezc, ezi[i],
                            typec, type2[i], adot_phi, adot_vel,
                            logr, xi, zeta_p, zeta_u,
                            psir_pp, psit_pp, psir_pu, psit_pu,
                            psir_uu, psit_uu, boxsize)
        field[i] = np.dot(cc1, inv_mat.dot(cc2))
        if mpi_rank == 0:
            progress_bar(i + 1, lenxi, lenpro, prefix)

    return field
