import numpy as np
from numba import njit

from .coords import distance_3d_float, get_vec_norm_float
from .interp import interp_log_float


@njit
def get_dd_float(x1, x2, y1, y2, z1, z2, logr, xi, boxsize):
    """Compute the density-density correlation for two points.

    Parameters
    ----------
    x1, y1, z1 : float
        Coordinates of point 1.
    x2, y2, z2 : float
        Coordinates of point 2.
    logr : array
        Logarithmic radius grid.
    xi : array
        Density correlation values on the grid.
    boxsize : float
        Periodic box size.

    Returns
    -------
    float
        Density-density correlation.
    """
    rx = x2 - x1
    ry = y2 - y1
    rz = z2 - z1
    newr, newrx, newry, newrz = distance_3d_float(rx, ry, rz, boxsize)
    return interp_log_float(logr, xi, np.log10(newr), xi[0], xi[-1])


@njit
def get_dp_float(x1, x2, y1, y2, z1, z2, ex, ey, ez, adot, logr, zeta, boxsize):
    """Compute the density-displacement/velocity cross-correlation.

    Parameters
    ----------
    x1, y1, z1 : float
        Coordinates of the first point.
    x2, y2, z2 : float
        Coordinates of the second point.
    ex, ey, ez : float
        Constraint direction vector.
    adot : float
        Time derivative factor.
    logr : array
        Logarithmic radius grid.
    zeta : array
        Cross-correlation values on the grid.
    boxsize : float
        Periodic box size.

    Returns
    -------
    float
        Density-displacement/velocity cross-correlation value.
    """
    rx = x2 - x1
    ry = y2 - y1
    rz = z2 - z1
    newr, newrx, newry, newrz = distance_3d_float(rx, ry, rz, boxsize)
    nrx, nry, nrz = get_vec_norm_float(newrx, newry, newrz)
    zeta_val = interp_log_float(logr, zeta, np.log10(newr), zeta[0], zeta[-1])
    return -adot*zeta_val*(ex*nrx + ey*nry + ez*nrz)


@njit
def get_pd_float(x1, x2, y1, y2, z1, z2, ex, ey, ez, adot, logr, zeta, boxsize):
    """Compute the density-displacement/velocity cross-correlation.

    Parameters
    ----------
    x1, y1, z1 : float
        Coordinates of the first point.
    x2, y2, z2 : float
        Coordinates of the second point.
    ex, ey, ez : float
        Constraint direction vector.
    adot : float
        Time derivative factor.
    logr : array
        Logarithmic radius grid.
    zeta : array
        Cross-correlation values on the grid.
    boxsize : float
        Periodic box size.

    Returns
    -------
    float
        Density-displacement/velocity cross-correlation value.
    """
    rx = x1 - x2
    ry = y1 - y2
    rz = z1 - z2
    newr, newrx, newry, newrz = distance_3d_float(rx, ry, rz, boxsize)
    nrx, nry, nrz = get_vec_norm_float(newrx, newry, newrz)
    zeta_val = interp_log_float(logr, zeta, np.log10(newr), zeta[0], zeta[-1])
    return -adot*zeta_val*(ex*nrx + ey*nry + ez*nrz)


@njit
def get_pp_float(
    x1, x2, y1, y2, z1, z2, 
    ex1, ex2, ey1, ey2, ez1, ez2,
    adot2, logr, psir, psit, boxsize
):
    """Compute the pairwise displacement/velocity autocorrelation.

    Parameters
    ----------
    x1, y1, z1 : float
        Coordinates of point 1.
    x2, y2, z2 : float
        Coordinates of point 2.
    ex1, ey1, ez1 : float
        Direction vector for point 1.
    ex2, ey2, ez2 : float
        Direction vector for point 2.
    adot2 : float
        Product of scaling factors for the correlation.
    logr : array
        Logarithmic radius grid.
    psir, psit : arrays
        Radial and tangential correlation profiles.
    boxsize : float
        Periodic box size.

    Returns
    -------
    float
        Pairwise displacement/velocity autocorrelation.
    """
    rx = x2 - x1
    ry = y2 - y1
    rz = z2 - z1

    newr, newrx, newry, newrz = distance_3d_float(rx, ry, rz, boxsize)
    nrx, nry, nrz = get_vec_norm_float(newrx, newry, newrz)

    pr = interp_log_float(logr, psir, np.log10(newr), psir[0], psir[-1])
    pt = interp_log_float(logr, psit, np.log10(newr), psit[0], psit[-1])

    p1 = pt
    p2 = pr - pt

    pp_xx = p1 + p2*nrx*nrx
    pp_yy = p1 + p2*nry*nry
    pp_zz = p1 + p2*nrz*nrz

    pp_xy = p2*nrx*nry
    pp_xz = p2*nrx*nrz
    pp_yx = p2*nry*nrx
    pp_yz = p2*nry*nrz
    pp_zx = p2*nrz*nrx
    pp_zy = p2*nrz*nry

    pp_x = ex1*(pp_xx*ex2 + pp_xy*ey2 + pp_xz*ez2)
    pp_y = ey1*(pp_yx*ex2 + pp_yy*ey2 + pp_yz*ez2)
    pp_z = ez1*(pp_zx*ex2 + pp_zy*ey2 + pp_zz*ez2)
    pp_val = adot2*(pp_x + pp_y + pp_z)

    if newr == 0.0:
        pp_val = adot2*pt*(ex1*ex2 + ey1*ey2 + ez1*ez2)
    return pp_val


@njit
def get_cc_float(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    type1, type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u,
    psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu, boxsize
):
    """Compute the full correlation between two points for a given type pair.

    Parameters
    ----------
    x1, y1, z1 : float
        Coordinates of point 1.
    x2, y2, z2 : float
        Coordinates of point 2.
    ex1, ey1, ez1 : float
        Direction vector for point 1 constraints.
    ex2, ey2, ez2 : float
        Direction vector for point 2 constraints.
    type1, type2 : int
        Constraint/correlation types.
    adot_phi, adot_vel : float
        Scaling factors for displacements and velocities.
    logr, xi, zeta_p, zeta_u, psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu : arrays
        Correlation profiles.
    boxsize : float
        Periodic box size.

    Returns
    -------
    float
        Correlation value for the specified type pair.
    """
    if type1 == 0 and type2 == 0:
        return get_dd_float(x1, x2, y1, y2, z1, z2, logr, xi, boxsize)
    if type1 == 1 and type2 == 1:
        return get_pp_float(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
                             adot_phi * adot_phi, logr, psir_pp, psit_pp, boxsize)
    if type1 == 2 and type2 == 2:
        return get_pp_float(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
                             adot_vel * adot_vel, logr, psir_uu, psit_uu, boxsize)
    if type1 == 0 and type2 == 1:
        return get_dp_float(x1, x2, y1, y2, z1, z2, ex2, ey2, ez2, adot_phi,
                             logr, zeta_p, boxsize)
    if type1 == 1 and type2 == 0:
        return get_pd_float(x1, x2, y1, y2, z1, z2, ex1, ey1, ez1, adot_phi,
                             logr, zeta_p, boxsize)
    if type1 == 0 and type2 == 2:
        return get_dp_float(x1, x2, y1, y2, z1, z2, ex2, ey2, ez2, adot_vel,
                             logr, zeta_u, boxsize)
    if type1 == 2 and type2 == 0:
        return get_pd_float(x1, x2, y1, y2, z1, z2, ex1, ey1, ez1, adot_vel,
                             logr, zeta_u, boxsize)
    if (type1 == 1 and type2 == 2) or (type1 == 2 and type2 == 1):
        return get_pp_float(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
                             adot_phi * adot_vel, logr, psir_pu, psit_pu, boxsize)
    return 0.0


@njit
def get_cc_array1(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    type1, type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u,
    psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu, boxsize
):
    """Compute correlations for an array of point 1 values against a single point 2.

    Parameters
    ----------
    x1, y1, z1 : array
        Coordinates of point 1 values.
    x2, y2, z2 : float
        Coordinates of the second point.
    ex1, ey1, ez1 : array
        Direction vectors for point 1 values.
    ex2, ey2, ez2 : float
        Direction vector for point 2.
    type1 : array
        Types for point 1 values.
    type2 : int
        Type for point 2.
    adot_phi, adot_vel : float
        Scaling factors.
    logr, xi, zeta_p, zeta_u, psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu : arrays
        Correlation profiles.
    boxsize : float
        Periodic box size.

    Returns
    -------
    array
        Correlation values for each point in point 1.
    """
    lenx1 = x1.shape[0]
    cc = np.empty(lenx1, dtype=np.float64)
    for i in range(lenx1):
        cc[i] = get_cc_float(x1[i], x2, y1[i], y2, z1[i], z2,
                             ex1[i], ex2, ey1[i], ey2, ez1[i], ez2,
                             type1[i], type2, adot_phi, adot_vel,
                             logr, xi, zeta_p, zeta_u,
                             psir_pp, psit_pp, psir_pu, psit_pu,
                             psir_uu, psit_uu, boxsize)
    return cc


@njit
def get_cc_array2(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    type1, type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u,
    psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu, boxsize
):
    """Compute correlations for a single point 1 against an array of point 2 values.

    Parameters
    ----------
    x1, y1, z1 : float
        Coordinates of the first point.
    x2, y2, z2 : array
        Coordinates of the second point values.
    ex1, ey1, ez1 : float
        Direction vector for point 1.
    ex2, ey2, ez2 : array
        Direction vectors for point 2 values.
    type1 : int
        Type for point 1.
    type2 : array
        Types for point 2 values.
    adot_phi, adot_vel : float
        Scaling factors.
    logr, xi, zeta_p, zeta_u, psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu : arrays
        Correlation profiles.
    boxsize : float
        Periodic box size.

    Returns
    -------
    array
        Correlation values for each point in point 2.
    """
    lenx2 = x2.shape[0]
    cc = np.empty(lenx2, dtype=np.float64)
    for i in range(lenx2):
        cc[i] = get_cc_float(x1, x2[i], y1, y2[i], z1, z2[i],
                             ex1, ex2[i], ey1, ey2[i], ez1, ez2[i],
                             type1, type2[i], adot_phi, adot_vel,
                             logr, xi, zeta_p, zeta_u,
                             psir_pp, psit_pp, psir_pu, psit_pu,
                             psir_uu, psit_uu, boxsize)
    return cc


@njit
def get_cc_arrays(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    type1, type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u,
    psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu, boxsize
):
    """Compute correlations for matching arrays of points.

    Parameters
    ----------
    x1, y1, z1 : array
        Coordinates of point 1 values.
    x2, y2, z2 : array
        Coordinates of point 2 values.
    ex1, ey1, ez1 : array
        Direction vectors for point 1 values.
    ex2, ey2, ez2 : array
        Direction vectors for point 2 values.
    type1, type2 : array
        Type arrays for the two point sets.
    adot_phi, adot_vel : float
        Scaling factors.
    logr, xi, zeta_p, zeta_u, psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu : arrays
        Correlation profiles.
    boxsize : float
        Periodic box size.

    Returns
    -------
    array
        Correlation values for each matching pair.
    """
    lenx = x1.shape[0]
    cc = np.empty(lenx, dtype=np.float64)
    for i in range(lenx):
        cc[i] = get_cc_float(
            x1[i], x2[i], y1[i], y2[i], z1[i], z2[i],
            ex1[i], ex2[i], ey1[i], ey2[i], ez1[i], ez2[i],
            type1[i], type2[i], adot_phi, adot_vel,
            logr, xi, zeta_p, zeta_u, psir_pp, psit_pp, 
            psir_pu, psit_pu, psir_uu, psit_uu, boxsize
        )
    return cc
