import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import spherical_jn


def get_sinc(x):
    """Returns the sinc function sin(x)/x

    Parameters
    ----------
    x : float/array
        Values to return sinc function.
    """
    if np.isscalar(x) == True:
        if x == 0:
            sinc = 0.
        else:
            sinc = np.sin(x)/x
    else:
        sinc = np.zeros(len(x))
        condition = np.where(x != 0.)[0]
        sinc[condition] = np.sin(x[condition])/x[condition]
    return sinc


def pk2xi(r, kh, pk, kmin=None, kmax=None, kfactor=100, kbinsmax=100000, Rg=None):
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
    kbinsmax : int, optional
        Maximum kbins for the interpolated pk
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
            kbinsmax=100000, Rg=None):
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
    kbinsmax : int, optional
        Maximum kbins for the interpolated pk
    Rg : float, optional
        Gaussian smoothing scale.

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
        if kbins > kbinsmax:
            kbins = kbinsmax
        k = np.linspace(kmin, kmax, kbins)
        p = interp_PK(k)
        if fk is None:
            if np.isscalar(fk):
                f = fk
            else:
                assert len(fk) == len(kh), 'If fk is scale dependent it must match length of kh'
                interp_fk = interp1d(kh, fk, kind='cubic')
                f = interp_fk(k)
        else:
            f = 1.
        if Rg is None:
            zeta[i] = (1./(2.*np.pi**2.))*simps(f*k*p*spherical_jn(1, k*r[i]), k)
        else:
            Wg = np.exp(-0.5*(k*Rg)**2.)
            zeta[i] = (1./(2.*np.pi**2.))*simps(f*k*Wg*p*spherical_jn(1, k*r[i]), k)
    return zeta


def pk2psiR(r, kh, pk, fk=None, kmin=None, kmax=None, kfactor=100,
            kbinsmax=100000, Rg=None):
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
    kbinsmax : int, optional
        Maximum kbins for the interpolated pk
    Rg : float, optional
        Gaussian smoothing scale.

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
        if kbins > kbinsmax:
            kbins = kbinsmax
        k = np.linspace(kmin, kmax, kbins)
        p = interp_PK(k)
        if fk is None:
            if np.isscalar(fk):
                f = fk
            else:
                assert len(fk) == len(kh), 'If fk is scale dependent it must match length of kh'
                interp_fk = interp1d(kh, fk, kind='cubic')
                f = interp_fk(k)
        else:
            f = 1.
        if Rg is None:
            psiR[i] = (1./(2.*np.pi**2.))*simps(f*f*p*(spherical_jn(0, k*r[i]) - 2*spherical_jn(1, k*r[i])/(k*r[i])), k)
        else:
            Wg = np.exp(-0.5*(k*Rg)**2.)
            psiR[i] = (1./(2.*np.pi**2.))*simps(f*f*Wg*p*(spherical_jn(0, k*r[i]) - 2*spherical_jn(1, k*r[i])/(k*r[i])), k)
    return psiR


def pk2psiT(r, kh, pk, fk=None, kmin=None, kmax=None, kfactor=100,
            kbinsmax=100000, Rg=None):
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
    kbinsmax : int, optional
        Maximum kbins for the interpolated pk
    Rg : float, optional
        Gaussian smoothing scale.

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
        if kbins > kbinsmax:
            kbins = kbinsmax
        k = np.linspace(kmin, kmax, kbins)
        p = interp_PK(k)
        if fk is None:
            if np.isscalar(fk):
                f = fk
            else:
                assert len(fk) == len(kh), 'If fk is scale dependent it must match length of kh'
                interp_fk = interp1d(kh, fk, kind='cubic')
                f = interp_fk(k)
        else:
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


def get_cov_du(rx, ry, rz, interp_zeta, z, interp_Hz):
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

    Returns
    -------
    cov_du : array_like
        Covariance overdensity to velocity cross-correlation.
    """
    r = np.sqrt(rx**2. + ry**2. + rz**2.)
    a = 1./(1.+z)
    adot = a*interp_Hz(z)
    norm_rx = np.copy(rx)/r
    norm_ry = np.copy(ry)/r
    norm_rz = np.copy(rz)/r
    cov_du_x = -adot*interp_zeta(r)*norm_rx
    cov_du_y = -adot*interp_zeta(r)*norm_ry
    cov_du_z = -adot*interp_zeta(r)*norm_rz
    cov_du = [cov_du_x, cov_du_y, cov_du_z]
    return cov_du


def get_cov_uu(rx, ry, rz, interp_psiR, interp_psiT, z, interp_Hz):
    """Returns the covariance overdensity to velocity relation.

    Parameters
    ----------
    rx, ry, rz : array_like
        Distance between data points in the x, y, and z-axis.
    interp_psiR, interp_psiT : function
        Velocity to velocity radial and tangential correlation interpolation function.
    z : float
        Redshift.
    interp_Hz : function
        Hubble interpolation function.

    Returns
    -------
    cov_uu : array_like
        Covariance velocity to velocity cross-correlation.
    """
    r = np.sqrt(rx**2. + ry**2. + rz**2.)
    a = 1./(1.+z)
    adot = a*interp_Hz(z)
    norm_rx = np.copy(rx)/r
    norm_ry = np.copy(ry)/r
    norm_rz = np.copy(rz)/r
    cov_uu_xx = (adot**2.)*(interp_psiT(r) + (interp_psiR(r) - interp_psiT(r))*norm_rx*norm_rx)
    cov_uu_yy = (adot**2.)*(interp_psiT(r) + (interp_psiR(r) - interp_psiT(r))*norm_ry*norm_ry)
    cov_uu_zz = (adot**2.)*(interp_psiT(r) + (interp_psiR(r) - interp_psiT(r))*norm_rz*norm_rz)
    cov_uu_xy = (adot**2.)*(interp_psiR(r) - interp_psiT(r))*norm_rx*norm_ry
    cov_uu_xz = (adot**2.)*(interp_psiR(r) - interp_psiT(r))*norm_rx*norm_rz
    cov_uu_yz = (adot**2.)*(interp_psiR(r) - interp_psiT(r))*norm_ry*norm_rz
    cov_uu_yx = np.copy(cov_uu_xy)
    cov_uu_zx = np.copy(cov_uu_xz)
    cov_uu_zy = np.copy(cov_uu_yz)
    cov_uu = [[cov_uu_xx, cov_uu_xy, cov_uu_xz],
              [cov_uu_yx, cov_uu_yy, cov_uu_yz],
              [cov_uu_zx, cov_uu_zy, cov_uu_zz]]
    return cov_uu
