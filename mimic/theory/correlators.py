import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import spherical_jn
#from magpie.utils import isscalar


def isscalar(x):
    """More general isscalar function to prevent 0 dimensional numpy arrays
    from being misidentified as arrays even though they are actually scalar
    variables.
    """
    if type(x).__module__ == np.__name__:
        if len(x.shape) == 0:
            return True
        else:
            return False
    else:
        return np.isscalar(x)


def get_sinc(x):
    """Returns the sinc function sin(x)/x

    Parameters
    ----------
    x : float/array
        Values to return sinc function.
    """
    if isscalar(x) == True:
        if x == 0:
            sinc = 0.
        else:
            sinc = np.sin(x)/x
    else:
        sinc = np.zeros(len(x))
        condition = np.where(x != 0.)[0]
        sinc[condition] = np.sin(x[condition])/x[condition]
    return sinc


def pk2xi(r, kh, pk, kmin=None, kmax=None, kfactor=100,
          kbinsmin=1000, kbinsmax=100000, Rg=None):
    """Power spectrum to 2-point density-density correlation function xi(r).

    Parameters
    ----------
    r : array
        Real space comoving distance.
    kh : array
        K values for the power spectrum.
    pk : array
        Power spectrum.
    kmin : float, optional
        Minimum k for sigmaR integration.
    kmax : float, optional
        Maximum k for sigmaR integration.
    kfactor : int, optional
        Binning extra factor.
    kbinsmin, kbinsmax : int, optional
        Min and max kbins for the interpolated pk
    Rg : float, optional
        Gaussian smoothing scale.

    Returns
    -------
    xi : array
        Two point correlation function.
    """
    interp_PK = interp1d(kh, pk, kind='cubic')
    if kmin is None:
        kmin = kh.min()
    if kmax is None:
        kmax = kh.max()
    xi = np.zeros(len(r))
    for i in range(0, len(xi)):
        kscale = 2.*np.pi/r[i]
        kbins = kfactor*int((kmax - kmin)/kscale)
        if kbins < kbinsmin:
            kbins = kbinsmin
        if kbins > kbinsmax:
            kbins = kbinsmax
        k = np.linspace(kmin, kmax, kbins)
        p = interp_PK(k)
        if Rg is None:
            xi[i] = (1./(2.*np.pi**2.))*simps((k**2.)*p*spherical_jn(0, k*r[i]), k)
        else:
            Wg = np.exp(-0.5*(k*Rg)**2.)
            xi[i] = (1./(2.*np.pi**2.))*simps((k**2.)*Wg*p*spherical_jn(0, k*r[i]), k)
    return xi


def pk2zeta(r, kh, pk, fk=None, kmin=None, kmax=None, kfactor=100,
            kbinsmin=1000, kbinsmax=100000, Rg=None, cons_type="Vel"):
    """Power spectrum to 2-point density-velocity correlation function xi(r).

    Parameters
    ----------
    r : array
        Real space comoving distance.
    kh : array
        K values for the power spectrum.
    pk : array
        Power spectrum.
    fk : scalar or array, optional
        Scale (in)dependent growth function. If dependent then fk must be the
        same length as kh and pk.
    kmin : float, optional
        Minimum k for sigmaR integration.
    kmax : float, optional
        Maximum k for sigmaR integration.
    kfactor : int, optional
        Binning extra factor.
    kbinsmin, kbinsmax : int, optional
        Min and max kbins for the interpolated pk
    Rg : float, optional
        Gaussian smoothing scale.
    cons_type : str
        Velocity or displacement field constraints.

    Returns
    -------
    zeta : array
        Two point density-velocity correlation function. If fk is given then
        this returns zeta_bar.
    """
    interp_PK = interp1d(kh, pk, kind='cubic')
    if kmin is None:
        kmin = kh.min()
    if kmax is None:
        kmax = kh.max()
    zeta = np.zeros(len(r))
    for i in range(0, len(zeta)):
        kscale = 2.*np.pi/r[i]
        kbins = kfactor*int((kmax - kmin)/kscale)
        if kbins < kbinsmin:
            kbins = kbinsmin
        if kbins > kbinsmax:
            kbins = kbinsmax
        k = np.linspace(kmin, kmax, kbins)
        p = interp_PK(k)
        if cons_type == "Vel":
            if fk is not None:
                if isscalar(fk):
                    f = fk
                else:
                    assert len(fk) == len(kh), 'If fk is scale dependent it must match length of kh'
                    interp_fk = interp1d(kh, fk, kind='cubic')
                    f = interp_fk(k)
            else:
                f = 1.
        elif cons_type == "Psi":
            f = 1.
        if Rg is None:
            zeta[i] = (1./(2.*np.pi**2.))*simps(f*k*p*spherical_jn(1, k*r[i]), k)
        else:
            Wg = np.exp(-0.5*(k*Rg)**2.)
            zeta[i] = (1./(2.*np.pi**2.))*simps(f*k*Wg*p*spherical_jn(1, k*r[i]), k)
    return zeta


def pk2psiR(r, kh, pk, fk=None, kmin=None, kmax=None, kfactor=100,
            kbinsmin=1000, kbinsmax=100000, Rg=None, cons_type="Vel"):
    """Power spectrum to 2-point radial velocity-velocity correlation function psiR(r).

    Parameters
    ----------
    r : array
        Real space comoving distance.
    kh : array
        K values for the power spectrum.
    pk : array
        Power spectrum.
    fk : scalar or array, optional
        Scale (in)dependent growth function. If dependent then fk must be the
        same length as kh and pk.
    kmin : float, optional
        Minimum k for sigmaR integration.
    kmax : float, optional
        Maximum k for sigmaR integration.
    kfactor : int, optional
        Binning extra factor.
    kbinsmin, kbinsmax : int, optional
        Min and max kbins for the interpolated pk
    Rg : float, optional
        Gaussian smoothing scale.
    cons_type : str
        Velocity or displacement field constraints.

    Returns
    -------
    psiR : array
        Two point radial velocity-velocity correlation function.
    """
    interp_PK = interp1d(kh, pk, kind='cubic')
    if kmin is None:
        kmin = kh.min()
    if kmax is None:
        kmax = kh.max()
    psiR = np.zeros(len(r))
    for i in range(0, len(psiR)):
        kscale = 2.*np.pi/r[i]
        kbins = kfactor*int((kmax - kmin)/kscale)
        if kbins < kbinsmin:
            kbins = kbinsmin
        if kbins > kbinsmax:
            kbins = kbinsmax
        k = np.linspace(kmin, kmax, kbins)
        p = interp_PK(k)
        if cons_type == "Vel":
            if fk is not None:
                if isscalar(fk):
                    f = fk
                else:
                    assert len(fk) == len(kh), 'If fk is scale dependent it must match length of kh'
                    interp_fk = interp1d(kh, fk, kind='cubic')
                    f = interp_fk(k)
            else:
                f = 1.
        elif cons_type == "Psi":
            f = 1.
        if Rg is None:
            psiR[i] = (1./(2.*np.pi**2.))*simps(f*f*p*(spherical_jn(0, k*r[i]) - 2*spherical_jn(1, k*r[i])/(k*r[i])), k)
        else:
            Wg = np.exp(-0.5*(k*Rg)**2.)
            psiR[i] = (1./(2.*np.pi**2.))*simps(f*f*Wg*p*(spherical_jn(0, k*r[i]) - 2*spherical_jn(1, k*r[i])/(k*r[i])), k)
    return psiR


def pk2psiT(r, kh, pk, fk=None, kmin=None, kmax=None, kfactor=100,
            kbinsmin=1000, kbinsmax=100000, Rg=None, cons_type="Vel"):
    """Power spectrum to 2-point transverse velocity-velocity correlation function psiT(r).

    Parameters
    ----------
    r : array
        Real space comoving distance.
    kh : array
        K values for the power spectrum.
    pk : array
        Power spectrum.
    fk : scalar or array, optional
        Scale (in)dependent growth function. If dependent then fk must be the
        same length as kh and pk.
    kmin : float, optional
        Minimum k for sigmaR integration.
    kmax : float, optional
        Maximum k for sigmaR integration.
    kfactor : int, optional
        Binning extra factor.
    kbinsmin, kbinsmax : int, optional
        Min and max kbins for the interpolated pk
    Rg : float, optional
        Gaussian smoothing scale.
    cons_type : str
        Velocity or displacement field constraints.

    Returns
    -------
    psiT : array
        Two point transverse velocity-velocity correlation function.
    """
    interp_PK = interp1d(kh, pk, kind='cubic')
    if kmin is None:
        kmin = kh.min()
    if kmax is None:
        kmax = kh.max()
    psiT = np.zeros(len(r))
    for i in range(0, len(psiT)):
        kscale = 2.*np.pi/r[i]
        kbins = kfactor*int((kmax - kmin)/kscale)
        if kbins < kbinsmin:
            kbins = kbinsmin
        if kbins > kbinsmax:
            kbins = kbinsmax
        k = np.linspace(kmin, kmax, kbins)
        p = interp_PK(k)
        if cons_type == "Vel":
            if fk is not None:
                if isscalar(fk):
                    f = fk
                else:
                    assert len(fk) == len(kh), 'If fk is scale dependent it must match length of kh'
                    interp_fk = interp1d(kh, fk, kind='cubic')
                    f = interp_fk(k)
            else:
                f = 1.
        elif cons_type == "Psi":
            f = 1.
        if Rg is None:
            psiT[i] = (1./(2.*np.pi**2.))*simps(f*f*p*spherical_jn(1, k*r[i])/(k*r[i]), k)
        else:
            Wg = np.exp(-0.5*(k*Rg)**2.)
            psiT[i] = (1./(2.*np.pi**2.))*simps(f*f*Wg*p*spherical_jn(1, k*r[i])/(k*r[i]), k)
    return psiT


def get_cov_dd(r, interp_xi):
    """Returns the covariance overdensity to overdensity relation.

    Parameters
    ----------
    r : array_like
        Distance between data points.
    interp_xi : function
        Overdensity auto-correlation interpolation function.
    """
    return interp_xi(r)


def periodic_1D(rx, boxsize):
    if np.isscalar(rx):
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


def periodic_3D(rx, ry, rz, boxsize):
    rx = periodic_1D(rx, boxsize)
    ry = periodic_1D(ry, boxsize)
    rz = periodic_1D(rz, boxsize)
    return rx, ry, rz


# def snap2grid1D(x, dx):
#     ind = np.around(x/dx)
#     return ind*dx
#
#
# def snap2grid3D(x, y, z, dx):
#     x = snap2grid1D(x, dx)
#     y = snap2grid1D(y, dx)
#     z = snap2grid1D(z, dx)
#     return x, y, z


def get_cov_dd_periodic(rx, ry, rz, boxsize, interp_xi):
    """Returns the covariance overdensity to overdensity relation.

    Parameters
    ----------
    rx, ry, rz : array_like
        Distance between data points along each axis.
    boxsize : float
        Size of the periodic box.
    interp_xi : function
        Overdensity auto-correlation interpolation function.
    """
    rx, ry, rz = periodic_3D(rx, ry, rz, boxsize)
    #dx = boxsize/ngrid
    #rx, ry, rz = snap2grid3D(rx, ry, rz, dx)
    r = np.sqrt(rx**2. + ry**2. + rz**2)
    return interp_xi(r)



def get_cov_du(rx, ry, rz, interp_zeta, z, interp_Hz, cons_type):
    """Returns the covariance overdensity to velocity relation.

    Parameters
    ----------
    rx, ry, rz : array_like
        Distance between data points in the x, y, and z-axis.
    interp_zeta : function
        Overdensity to velocity cross-correlation interpolation function.
    z : float
        Redshift.
    interp_Hz : function
        Hubble interpolation function.
    cons_type : str
        Velocity or displacement field constraints.

    Returns
    -------
    cov_du : array_like
        Covariance overdensity to velocity cross-correlation.
    """
    r = np.sqrt(rx**2. + ry**2. + rz**2.)
    a = 1./(1.+z)
    if cons_type == "Vel":
        adot = a*interp_Hz(z)
    elif cons_type == "Psi":
        adot = a
    norm_rx = np.copy(rx)/r
    norm_ry = np.copy(ry)/r
    norm_rz = np.copy(rz)/r
    cov_du = interp_zeta(r)
    cov_du_x = -adot*cov_du*norm_rx
    cov_du_y = -adot*cov_du*norm_ry
    cov_du_z = -adot*cov_du*norm_rz
    cov_du = [cov_du_x, cov_du_y, cov_du_z]
    return cov_du


def get_cov_du_periodic(rx, ry, rz, boxsize, interp_zeta, z, interp_Hz, cons_type):
    """Returns the covariance overdensity to velocity relation.

    Parameters
    ----------
    rx, ry, rz : array_like
        Distance between data points in the x, y, and z-axis.
    boxsize : float
        Size of the periodic box.
    interp_zeta : function
        Overdensity to velocity cross-correlation interpolation function.
    z : float
        Redshift.
    interp_Hz : function
        Hubble interpolation function.
    cons_type : str
        Velocity or displacement field constraints.

    Returns
    -------
    cov_du : array_like
        Covariance overdensity to velocity cross-correlation.
    """
    rx, ry, rz = periodic_3D(rx, ry, rz, boxsize)
    #dx = boxsize/ngrid
    #rx, ry, rz = snap2grid3D(rx, ry, rz, dx)
    r = np.sqrt(rx**2. + ry**2. + rz**2.)
    a = 1./(1.+z)
    if cons_type == "Vel":
        adot = a*interp_Hz(z)
    elif cons_type == "Psi":
        adot = a
    norm_rx = np.copy(rx)/r
    norm_ry = np.copy(ry)/r
    norm_rz = np.copy(rz)/r
    cov_du = interp_zeta(r)
    cov_du_x = -adot*cov_du*norm_rx
    cov_du_y = -adot*cov_du*norm_ry
    cov_du_z = -adot*cov_du*norm_rz
    cov_du = [cov_du_x, cov_du_y, cov_du_z]
    return cov_du


def get_cov_uu(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    interp_psiR, interp_psiT, z, interp_Hz, psiT0, cons_type):
    """Returns the covariance overdensity to velocity relation.

    Parameters
    ----------
    x1, y1, z1 : array_like
        Positions of constraints 1.
    x2, y2, z2 : array_like
        Positions of constraints 2.
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
    cons_type : str
        Velocity or displacement field constraints.

    Returns
    -------
    cov_uu : array_like
        Covariance velocity to velocity cross-correlation.
    """
    r = np.sqrt((x2-x1)**2. + (y2-y1)**2. + (z2-z1)**2.)
    a = 1./(1.+z)
    if cons_type == "Vel":
        adot = a*interp_Hz(z)
    elif cons_type == "Psi":
        adot = a
    cond = np.where(r != 0.)
    nx = np.zeros(np.shape(r))
    ny = np.zeros(np.shape(r))
    nz = np.zeros(np.shape(r))
    nx[cond] = (x2[cond]-x1[cond])/r[cond]
    ny[cond] = (y2[cond]-y1[cond])/r[cond]
    nz[cond] = (z2[cond]-z1[cond])/r[cond]
    nx1, nx2 = nx, nx
    ny1, ny2 = ny, ny
    nz1, nz2 = nz, nz
    cov_uu_ii = (adot**2.)*interp_psiT(r)
    cov_uu_jj = (adot**2.)*(interp_psiR(r) - interp_psiT(r))
    cov_uu_xx = cov_uu_ii + cov_uu_jj*nx*nx
    cov_uu_yy = cov_uu_ii + cov_uu_jj*ny*ny
    cov_uu_zz = cov_uu_ii + cov_uu_jj*nz*nz
    cov_uu_xy = cov_uu_jj*nx*ny
    cov_uu_xz = cov_uu_jj*nx*nz
    cov_uu_yz = cov_uu_jj*ny*nz
    cov_uu_yx = cov_uu_jj*ny*nx
    cov_uu_zx = cov_uu_jj*nz*nx
    cov_uu_zy = cov_uu_jj*nz*ny
    cov_uu = [[cov_uu_xx, cov_uu_xy, cov_uu_xz],
              [cov_uu_yx, cov_uu_yy, cov_uu_yz],
              [cov_uu_zx, cov_uu_zy, cov_uu_zz]]
    cov_uu  = (cov_uu_xx*ex2 + cov_uu_xy*ey2 + cov_uu_xz*ez2)*ex1
    cov_uu += (cov_uu_yx*ex2 + cov_uu_yy*ey2 + cov_uu_yz*ez2)*ey1
    cov_uu += (cov_uu_zx*ex2 + cov_uu_zy*ey2 + cov_uu_zz*ez2)*ez1
    cond = np.where(r == 0.)
    cov_uu[cond[0],cond[1]] = (adot**2.)*psiT0*(ex1[cond[0],cond[1]]*ex2[cond[0],cond[1]]+ey1[cond[0],cond[1]]*ey2[cond[0],cond[1]]+ez1[cond[0],cond[1]]*ez2[cond[0],cond[1]])
    return cov_uu



def get_cov_uu_periodic(x1, x2, y1, y2, z1, z2, boxsize, ex1, ex2, ey1, ey2, ez1, ez2,
    interp_psiR, interp_psiT, z, interp_Hz, psiT0, cons_type):
    """Returns the covariance overdensity to velocity relation.

    Parameters
    ----------
    x1, y1, z1 : array_like
        Positions of constraints 1.
    x2, y2, z2 : array_like
        Positions of constraints 2.
    boxsize : float
        Size of the periodic box.
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
    cons_type : str
        Velocity or displacement field constraints.

    Returns
    -------
    cov_uu : array_like
        Covariance velocity to velocity cross-correlation.
    """
    rx = x2 - x1
    ry = y2 - y1
    rz = z2 - z1
    rx, ry, rz = periodic_3D(rx, ry, rz, boxsize)
    #dx = boxsize/ngrid
    #rx, ry, rz = snap2grid3D(rx, ry, rz, dx)
    r = np.sqrt(rx**2. + ry**2. + rz**2.)
    a = 1./(1.+z)
    if cons_type == "Vel":
        adot = a*interp_Hz(z)
    elif cons_type == "Psi":
        adot = a
    cond = np.where(r != 0.)
    nx = np.zeros(np.shape(r))
    ny = np.zeros(np.shape(r))
    nz = np.zeros(np.shape(r))
    nx[cond] = (x2[cond]-x1[cond])/r[cond]
    ny[cond] = (y2[cond]-y1[cond])/r[cond]
    nz[cond] = (z2[cond]-z1[cond])/r[cond]
    nx1, nx2 = nx, nx
    ny1, ny2 = ny, ny
    nz1, nz2 = nz, nz
    cov_uu_ii = (adot**2.)*interp_psiT(r)
    cov_uu_jj = (adot**2.)*(interp_psiR(r) - interp_psiT(r))
    cov_uu_xx = cov_uu_ii + cov_uu_jj*nx*nx
    cov_uu_yy = cov_uu_ii + cov_uu_jj*ny*ny
    cov_uu_zz = cov_uu_ii + cov_uu_jj*nz*nz
    cov_uu_xy = cov_uu_jj*nx*ny
    cov_uu_xz = cov_uu_jj*nx*nz
    cov_uu_yz = cov_uu_jj*ny*nz
    cov_uu_yx = cov_uu_jj*ny*nx
    cov_uu_zx = cov_uu_jj*nz*nx
    cov_uu_zy = cov_uu_jj*nz*ny
    cov_uu = [[cov_uu_xx, cov_uu_xy, cov_uu_xz],
              [cov_uu_yx, cov_uu_yy, cov_uu_yz],
              [cov_uu_zx, cov_uu_zy, cov_uu_zz]]
    cov_uu  = (cov_uu_xx*ex2 + cov_uu_xy*ey2 + cov_uu_xz*ez2)*ex1
    cov_uu += (cov_uu_yx*ex2 + cov_uu_yy*ey2 + cov_uu_yz*ez2)*ey1
    cov_uu += (cov_uu_zx*ex2 + cov_uu_zy*ey2 + cov_uu_zz*ez2)*ez1
    cond = np.where(r == 0.)
    cov_uu[cond[0],cond[1]] = (adot**2.)*psiT0*(ex1[cond[0],cond[1]]*ex2[cond[0],cond[1]]+ey1[cond[0],cond[1]]*ey2[cond[0],cond[1]]+ez1[cond[0],cond[1]]*ez2[cond[0],cond[1]])
    return cov_uu
