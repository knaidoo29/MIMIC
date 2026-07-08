import numpy as np
from numba import njit

from fiesta import src


# @njit
# def trilinear_float(
#     fgrid: np.ndarray,
#     x: float,
#     y: float,
#     z: float,
#     boxsize: float,
#     ngrid: int
# ) -> float:  # pragma: no cover
#     """
#     Trilinear interpolation of field defined on a grid.

#     Parameters
#     ----------
#     fgrid : array
#         Field values on the grid.
#     x : float
#         X coordinates where we need interpolated values.
#     y : float
#         Y coordinates where we need interpolated values.
#     z : float
#         Z coordinates where we need interpolated values.
#     boxsize : float
#         Size of the box.
#     ngrid : int
#         Size of the grid along each axis.
#     npart : int
#         Number of particles.

#     Returns
#     -------
#     f : array
#         Interpolated field values.
#     """
#     xp, yp, zp = x, y, z
#     minx = 0.0
#     dx = boxsize / float(ngrid)

#     if x - dx / 2.0 < 0.0:
#         xp = xp + boxsize

#     if yp - dx / 2.0 < 0.0:
#         yp = yp + boxsize

#     if zp - dx / 2.0 < 0.0:
#         zp = zp + boxsize

#     ix1 = int((xp - dx / 2.0) / dx)
#     xg1 = src.xgrid(ix1, dx, minx)

#     ix2 = ix1 + 1
#     xg2 = src.xgrid(ix2, dx, minx)

#     if ix2 == ngrid:
#         ix2 = ix2 - ngrid

#     iy1 = int((yp - dx / 2.0) / dx)
#     yg1 = src.xgrid(iy1, dx, minx)

#     iy2 = iy1 + 1
#     yg2 = src.xgrid(iy2, dx, minx)

#     if iy2 == ngrid:
#         iy2 = iy2 - ngrid

#     iz1 = int((zp - dx / 2.0) / dx)
#     zg1 = src.xgrid(iz1, dx, minx)

#     iz2 = iz1 + 1
#     zg2 = src.xgrid(iz2, dx, minx)

#     if iz2 == ngrid:
#         iz2 = iz2 - ngrid

#     # surround points in the grid of a single point for interpolation.

#     q111 = iz1 + ngrid * (iy1 + ngrid * ix1)
#     q112 = iz1 + ngrid * (iy1 + ngrid * ix2)
#     q121 = iz1 + ngrid * (iy2 + ngrid * ix1)
#     q122 = iz1 + ngrid * (iy2 + ngrid * ix2)
#     q211 = iz2 + ngrid * (iy1 + ngrid * ix1)
#     q212 = iz2 + ngrid * (iy1 + ngrid * ix2)
#     q221 = iz2 + ngrid * (iy2 + ngrid * ix1)
#     q222 = iz2 + ngrid * (iy2 + ngrid * ix2)

#     f111 = fgrid[q111]
#     f112 = fgrid[q112]
#     f121 = fgrid[q121]
#     f122 = fgrid[q122]
#     f211 = fgrid[q211]
#     f212 = fgrid[q212]
#     f221 = fgrid[q221]
#     f222 = fgrid[q222]

#     xd = (xp - xg1) / (xg2 - xg1)
#     yd = (yp - yg1) / (yg2 - yg1)
#     zd = (zp - zg1) / (zg2 - zg1)

#     f11 = f111 * (1 - xd) + f112 * xd
#     f21 = f211 * (1 - xd) + f212 * xd
#     f12 = f121 * (1 - xd) + f122 * xd
#     f22 = f221 * (1 - xd) + f222 * xd

#     f1 = f11 * (1 - yd) + f12 * yd
#     f2 = f21 * (1 - yd) + f22 * yd

#     f = f1 * (1 - zd) + f2 * zd

#     return f

@njit
def stretch_sin_backward(xcos, boxsize):
    return boxsize*(np.sin(np.pi*xcos - np.pi/2.)+1)/2.
@njit
def stretch_sin_forward(x, boxsize):
    return np.arcsin(2*x/boxsize-1.)/np.pi + 1./2.

@njit
def trilinear_float(fgrid, x, y, z, boxsize, ngrid, stretch):
    """Periodic trilinear interpolation on a flattened C-order cube.

    The grid is assumed to live at cell centres:
        x_i = (i + 1/2) * dx
    with periodic wrapping.
    """

    # Periodic coordinates in [0, boxsize)
    x = x % boxsize
    y = y % boxsize
    z = z % boxsize

    if stretch:
        x = stretch_sin_forward(x, boxsize)
        y = stretch_sin_forward(y, boxsize)
        z = stretch_sin_forward(z, boxsize)
        dx = 1. / float(ngrid)
    else:
        dx = boxsize / float(ngrid)

    # Coordinate relative to cell-centred grid.
    # u = 0 means exactly at centre of cell 0.
    ux = x / dx - 0.5
    uy = y / dx - 0.5
    uz = z / dx - 0.5

    ix0_raw = int(np.floor(ux))
    iy0_raw = int(np.floor(uy))
    iz0_raw = int(np.floor(uz))

    tx = ux - ix0_raw
    ty = uy - iy0_raw
    tz = uz - iz0_raw

    ix0 = ix0_raw % ngrid
    iy0 = iy0_raw % ngrid
    iz0 = iz0_raw % ngrid

    ix1 = (ix0 + 1) % ngrid
    iy1 = (iy0 + 1) % ngrid
    iz1 = (iz0 + 1) % ngrid

    # Flattened index for array[ix, iy, iz] in C-order.
    q000 = iz0 + ngrid * (iy0 + ngrid * ix0)
    q100 = iz0 + ngrid * (iy0 + ngrid * ix1)
    q010 = iz0 + ngrid * (iy1 + ngrid * ix0)
    q110 = iz0 + ngrid * (iy1 + ngrid * ix1)

    q001 = iz1 + ngrid * (iy0 + ngrid * ix0)
    q101 = iz1 + ngrid * (iy0 + ngrid * ix1)
    q011 = iz1 + ngrid * (iy1 + ngrid * ix0)
    q111 = iz1 + ngrid * (iy1 + ngrid * ix1)

    f000 = fgrid[q000]
    f100 = fgrid[q100]
    f010 = fgrid[q010]
    f110 = fgrid[q110]

    f001 = fgrid[q001]
    f101 = fgrid[q101]
    f011 = fgrid[q011]
    f111 = fgrid[q111]

    f00 = f000 * (1.0 - tx) + f100 * tx
    f10 = f010 * (1.0 - tx) + f110 * tx

    f01 = f001 * (1.0 - tx) + f101 * tx
    f11 = f011 * (1.0 - tx) + f111 * tx

    f0 = f00 * (1.0 - ty) + f10 * ty
    f1 = f01 * (1.0 - ty) + f11 * ty

    return f0 * (1.0 - tz) + f1 * tz

@njit
def get_dd_grid_float(x1, x2, y1, y2, z1, z2, xi_box, ngrid, boxsize, stretch):
    """Compute the density-density correlation for two points.

    Parameters
    ----------
    x1, y1, z1 : float
        Coordinates of point 1.
    x2, y2, z2 : float
        Coordinates of point 2.
    xi_box : array
        Density correlation values on the grid.
    ngrid : int
        Number of grid points along each axis.
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
    rx %= boxsize
    ry %= boxsize
    rz %= boxsize
    if rx == boxsize:
        rx = 0.
    if ry == boxsize:
        ry = 0.
    if rz == boxsize:
        rz = 0.
    xi = trilinear_float(xi_box, rx, ry, rz, boxsize, ngrid, stretch)
    return xi


@njit
def get_dp_grid_float(x1, x2, y1, y2, z1, z2, ex, ey, ez, adot, zeta_p_box, ngrid, boxsize, stretch):
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
    zeta_p_box : array
        Cross-correlation values on the grid.
    ngrid : int
        Number of grid points along each axis.
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
    rx %= boxsize
    ry %= boxsize
    rz %= boxsize
    if rx == boxsize:
        rx = 0.
    if ry == boxsize:
        ry = 0.
    if rz == boxsize:
        rz = 0.
    zeta_x = trilinear_float(zeta_p_box, rx, ry, rz, boxsize, ngrid, stretch)
    zeta_y = trilinear_float(zeta_p_box, ry, rz, rx, boxsize, ngrid, stretch)
    zeta_z = trilinear_float(zeta_p_box, rz, rx, ry, boxsize, ngrid, stretch)
    return -adot*(zeta_x*ex + zeta_y*ey + zeta_z*ez)


@njit
def get_pd_grid_float(x1, x2, y1, y2, z1, z2, ex, ey, ez, adot, zeta_p_box, ngrid, boxsize, stretch):
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
    zeta_p_box : array
        Cross-correlation values on the grid.
    Ngrid : int
        Number of grid points along each axis.
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
    rx %= boxsize
    ry %= boxsize
    rz %= boxsize
    if rx == boxsize:
        rx = 0.
    if ry == boxsize:
        ry = 0.
    if rz == boxsize:
        rz = 0.
    zeta_x = trilinear_float(zeta_p_box, rx, ry, rz, boxsize, ngrid, stretch)
    zeta_y = trilinear_float(zeta_p_box, ry, rz, rx, boxsize, ngrid, stretch)
    zeta_z = trilinear_float(zeta_p_box, rz, rx, ry, boxsize, ngrid, stretch)
    return -adot*(zeta_x*ex + zeta_y*ey + zeta_z*ez)


@njit
def get_pp_grid_float(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    adot2, psi_pp_xx, psi_pp_xy, ngrid, boxsize, stretch
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
    psi_pp_xx, psi_pp_xy : arrays
        Radial and tangential correlation profiles.
    ngrid : int
        Number of grid points along each axis.  
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
    rx %= boxsize
    ry %= boxsize
    rz %= boxsize
    if rx == boxsize:
        rx = 0.
    if ry == boxsize:
        ry = 0.
    if rz == boxsize:
        rz = 0.

    pp_xx = trilinear_float(psi_pp_xx, rx, ry, rz, boxsize, ngrid, stretch)
    pp_xy = trilinear_float(psi_pp_xy, rx, ry, rz, boxsize, ngrid, stretch)
    pp_xz = trilinear_float(psi_pp_xy, rx, rz, ry, boxsize, ngrid, stretch)

    pp_yx = trilinear_float(psi_pp_xy, ry, rx, rz, boxsize, ngrid, stretch)
    pp_yy = trilinear_float(psi_pp_xx, ry, rz, rx, boxsize, ngrid, stretch)
    pp_yz = trilinear_float(psi_pp_xy, ry, rz, rx, boxsize, ngrid, stretch)

    pp_zx = trilinear_float(psi_pp_xy, rz, rx, ry, boxsize, ngrid, stretch)
    pp_zy = trilinear_float(psi_pp_xy, rz, ry, rx, boxsize, ngrid, stretch)
    pp_zz = trilinear_float(psi_pp_xx, rz, rx, ry, boxsize, ngrid, stretch)
    
    pp_val =  ex2*(ex1*pp_xx + ey1*pp_yx + ez1*pp_zx)
    pp_val += ey2*(ex1*pp_xy + ey1*pp_yy + ez1*pp_zy)
    pp_val += ez2*(ex1*pp_xz + ey1*pp_yz + ez1*pp_zz)
    pp_val *= adot2

    return pp_val


@njit
def get_cc_grid_float(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    type1, type2, adot_phi, adot_vel,  xi_box, zeta_p_box, zeta_u_box,
    psi_pp_xx, psi_pp_xy, psi_pu_xx, psi_pu_xy, psi_uu_xx, psi_uu_xy, 
    ngrid, boxsize, stretch
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
    xi_box, zeta_p_box, zeta_u_box , psi_pp_xx, psi_pp_xy, psi_pu_xx, psi_pu_xy, psi_uu_xx, psi_uu_xy : arrays
        Correlation profiles.
    ngrid : int
        Number of grid points along each axis.
    boxsize : float
        Periodic box size.

    Returns
    -------
    float
        Correlation value for the specified type pair.
    """
    if type1 == 0 and type2 == 0:
        return get_dd_grid_float(x1, x2, y1, y2, z1, z2, xi_box, ngrid, boxsize, stretch)
    if type1 == 1 and type2 == 1:
        return get_pp_grid_float(
            x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
            adot_phi * adot_phi, psi_pp_xx, psi_pp_xy, ngrid,boxsize, stretch
        )
    if type1 == 2 and type2 == 2:
        return get_pp_grid_float(
            x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
            adot_vel * adot_vel, psi_uu_xx, psi_uu_xy, ngrid, boxsize, stretch
        )
    if type1 == 0 and type2 == 1:
        return get_dp_grid_float(
            x1, x2, y1, y2, z1, z2, ex2, ey2, ez2, adot_phi, zeta_p_box, ngrid, boxsize, stretch
        )
    if type1 == 1 and type2 == 0:
        return get_pd_grid_float(
            x1, x2, y1, y2, z1, z2, ex1, ey1, ez1, adot_phi, zeta_p_box, ngrid, boxsize, stretch
        )
    if type1 == 0 and type2 == 2:
        return get_dp_grid_float(
            x1, x2, y1, y2, z1, z2, ex2, ey2, ez2, adot_vel, zeta_u_box, ngrid, boxsize, stretch
        )
    if type1 == 2 and type2 == 0:
        return get_pd_grid_float(
            x1, x2, y1, y2, z1, z2, ex1, ey1, ez1, adot_vel, zeta_u_box, ngrid, boxsize, stretch
        )
    if (type1 == 1 and type2 == 2) or (type1 == 2 and type2 == 1):
        return get_pp_grid_float(
            x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
            adot_phi * adot_vel, psi_pu_xx, psi_pu_xy, ngrid, boxsize, stretch
        )
    return 0.0


@njit
def get_cc_grid_array1(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    type1, type2, adot_phi, adot_vel, xi_box, zeta_p_box, zeta_u_box,
    psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, 
    psixx_uu_box, psixy_uu_box, ngrid, boxsize, stretch
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
    xi_box, zeta_p_box, zeta_u_box , psi_pp_xx, psi_pp_xy, psi_pu_xx, psi_pu_xy, psi_uu_xx, psi_uu_xy : arrays
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
        cc[i] = get_cc_grid_float(
            x1[i], x2, y1[i], y2, z1[i], z2, ex1[i], ex2, ey1[i], ey2, ez1[i], ez2,
            type1[i], type2, adot_phi, adot_vel, xi_box, zeta_p_box, zeta_u_box,
            psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, 
            psixx_uu_box, psixy_uu_box, ngrid, boxsize, stretch
        )
    return cc


@njit
def get_cc_grid_array2(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    type1, type2, adot_phi, adot_vel, xi_box, zeta_p_box, zeta_u_box,
    psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, 
    psixx_uu_box, psixy_uu_box, ngrid, boxsize, stretch
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
    xi_box, zeta_p_box, zeta_u_box , psi_pp_xx, psi_pp_xy, psi_pu_xx, psi_pu_xy, psi_uu_xx, psi_uu_xy : arrays
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
        cc[i] = get_cc_grid_float(
            x1, x2[i], y1, y2[i], z1, z2[i], ex1, ex2[i], ey1, ey2[i], ez1, ez2[i],
            type1, type2[i], adot_phi, adot_vel, xi_box, zeta_p_box, zeta_u_box,
            psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, 
            psixx_uu_box, psixy_uu_box, ngrid, boxsize, stretch
        )
    return cc


@njit
def get_cc_grid_arrays(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    type1, type2, adot_phi, adot_vel, xi_box, zeta_p_box, zeta_u_box,
    psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, 
    psixx_uu_box, psixy_uu_box, ngrid, boxsize, stretch
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
    xi_box, zeta_p_box, zeta_u_box , psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box : arrays
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
        cc[i] = get_cc_grid_float(
            x1[i], x2[i], y1[i], y2[i], z1[i], z2[i],
            ex1[i], ex2[i], ey1[i], ey2[i], ez1[i], ez2[i],
            type1[i], type2[i], adot_phi, adot_vel,
            xi_box, zeta_p_box, zeta_u_box, psixx_pp_box, psixy_pp_box, 
            psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box, ngrid, boxsize, stretch
        )
    return cc
