import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import spherical_jn


from ..io import isscalar


def pk2xi(r, kh, pk, kmin=None, kmax=None, kfactor=100, kbinsmin=1000,
    kbinsmax=100000, Rg=None):
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


def pk2zeta(r, kh, pk, fk=None, kmin=None, kmax=None, kfactor=100, kbinsmin=1000,
    kbinsmax=100000, Rg=None, cons_type="Vel"):
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
        if fk is not None:
            if isscalar(fk):
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


def pk2psiR(r, kh, pk, fk=None, kmin=None, kmax=None, kfactor=100, kbinsmin=1000,
    kbinsmax=100000, Rg=None, cons_type="Vel"):
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
        if fk is not None:
            if isscalar(fk):
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


def pk2psiT(r, kh, pk, fk=None, kmin=None, kmax=None, kfactor=100, kbinsmin=1000,
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
    kbinsmin, kbinsmax : int, optional
        Min and max kbins for the interpolated pk
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
        if kbins < kbinsmin:
            kbins = kbinsmin
        if kbins > kbinsmax:
            kbins = kbinsmax
        k = np.linspace(kmin, kmax, kbins)
        p = interp_PK(k)
        if fk is not None:
            if isscalar(fk):
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
