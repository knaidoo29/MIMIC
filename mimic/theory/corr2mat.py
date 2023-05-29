import numpy as np


from . import coords, Hz


def get_corr_dd(x1, x2, y1, y2, z1, z2, interp_xi, boxsize=None):
    """Returns the autocorrelation overdensity function.

    Parameters
    ----------
    x1, x2, y1, y2, z1, z2 : array/float
        Coordinates of points 1 and 2 in cartesian coordinates.
    interp_xi : function
        Overdensity auto-correlation interpolation function.
    boxsize : float, optional
        If provided, then periodic boundaries are assumed.
    """
    r = coords.distance_3D(x1, x2, y1, y2, z1, z2, boxsize=boxsize)
    return interp_xi(r)


def get_corr_du(x1, x2, y1, y2, z1, z2, interp_zeta, z, interp_Hz, boxsize=None,
    velocity=True):
    """Returns the cross correlation overdensity to velocity.

    Parameters
    ----------
    x1, x2, y1, y2, z1, z2 : array/float
        Coordinates of points 1 and 2 in cartesian coordinates.
    interp_zeta : function
        Overdensity to velocity cross-correlation interpolation function.
    z : float
        Redshift.
    interp_Hz : function
        Hubble interpolation function.
    boxsize : float, optional
        If provided, then periodic boundaries are assumed.
    velocity : bool, optional
        If True then output is given in velocity units. Note if not true then it
        is important that interp_zeta does not include the growth rate in its
        calculation.

    Returns
    -------
    corr_du : array_like
        Overdensity to velocity cross-correlation.
    """
    rx, ry, rz, r = coords.distance_3D(x1, x2, y1, y2, z1, z2,
        boxsize=boxsize, return_axis_dist=True)
    a = Hz.z2a(z)
    if velocity:
        adot = a*interp_Hz(z)
    else:
        adot = a
    norm_rx = np.copy(rx)/r
    norm_ry = np.copy(ry)/r
    norm_rz = np.copy(rz)/r
    corr_du = interp_zeta(r)
    corr_du_x = -adot*corr_du*norm_rx
    corr_du_y = -adot*corr_du*norm_ry
    corr_du_z = -adot*corr_du*norm_rz
    corr_du = [cov_du_x, cov_du_y, cov_du_z]
    return corr_du


def get_corr_uu(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    interp_psiR, interp_psiT, z, interp_Hz, psiT0, cons_type=None, boxsize=None,
    velocity=True):
    """Returns the covariance overdensity to velocity relation.

    Parameters
    ----------
    x1, x2, y1, y2, z1, z2 : array/float
        Coordinates of points 1 and 2 in cartesian coordinates.
    ex1, ey1, ez1 : array_like
        Unit vector of constraints 1 peculiar velocity.
    ex2, ey2, ez2 : array_like
        Unit vector of constraints 1 peculiar velocity.
    interp_psiR, interp_psiT : function
        Velocity to velocity radial and tangential correlation interpolation function.
    z : float
        Redshift.
    interp_Hz : function
        Hubble interpolation function.
    PsiT0 : float
        Value of PsiT at r=0.
    velocity : bool, optional
        If True then output is given in velocity units. Note if not true then it
        is important that interp_zeta does not include the growth rate in its
        calculation.

    Returns
    -------
    corr_uu : array_like
        Velocity to velocity cross-correlation matrix.
    """
    rx, ry, rz, r = coords.distance_3D(x1, x2, y1, y2, z1, z2,
        boxsize=boxsize, return_axis_dist=True)
    a = Hz.z2a(z)
    if velocity:
        adot = a*interp_Hz(z)
    else:
        adot = a
    cond = np.where(r != 0.)
    nx = np.zeros(np.shape(r))
    ny = np.zeros(np.shape(r))
    nz = np.zeros(np.shape(r))
    nx[cond] = rx[cond]/r[cond]
    ny[cond] = ry[cond]/r[cond]
    nz[cond] = rz[cond]/r[cond]
    nx1, nx2 = nx, nx
    ny1, ny2 = ny, ny
    nz1, nz2 = nz, nz
    corr_uu_ii = (adot**2.)*interp_psiT(r)
    corr_uu_jj = (adot**2.)*(interp_psiR(r) - interp_psiT(r))
    corr_uu_xx = corr_uu_ii + corr_uu_jj*nx*nx
    corr_uu_yy = corr_uu_ii + corr_uu_jj*ny*ny
    corr_uu_zz = corr_uu_ii + corr_uu_jj*nz*nz
    corr_uu_xy = corr_uu_jj*nx*ny
    corr_uu_xz = corr_uu_jj*nx*nz
    corr_uu_yz = corr_uu_jj*ny*nz
    corr_uu_yx = corr_uu_jj*ny*nx
    corr_uu_zx = corr_uu_jj*nz*nx
    corr_uu_zy = corr_uu_jj*nz*ny
    corr_uu = [[corr_uu_xx, corr_uu_xy, corr_uu_xz],
               [corr_uu_yx, corr_uu_yy, corr_uu_yz],
               [corr_uu_zx, corr_uu_zy, corr_uu_zz]]
    corr_uu  = (corr_uu_xx*ex2 + corr_uu_xy*ey2 + corr_uu_xz*ez2)*ex1
    corr_uu += (corr_uu_yx*ex2 + corr_uu_yy*ey2 + corr_uu_yz*ez2)*ey1
    corr_uu += (corr_uu_zx*ex2 + corr_uu_zy*ey2 + corr_uu_zz*ez2)*ez1
    cond = np.where(r == 0.)
    corr_uu[cond[0],cond[1]] = (adot**2.)*psiT0*(ex1[cond[0],cond[1]]*ex2[cond[0],cond[1]]+ey1[cond[0],cond[1]]*ey2[cond[0],cond[1]]+ez1[cond[0],cond[1]]*ez2[cond[0],cond[1]])
    return corr_uu
