import numpy as np


def _save_correlators_npz(fname, redshift, r, xi, zeta, psiR, psiT):
    """Save correlation functions.

    Parameters
    ----------
    fname : str
        Filename.
    redshift : float
        Redshift.
    r : array
        Radial axis.
    xi : array
        Density auto-correlation function.
    zeta : array
        Density-velocity cross correlation function.
    PsiR : array
        Velocity auto-correlation radial function.
    PsiT : array
        Velocity auto-correlation tangential function.
    """
    np.savez(fname, redshift=redshift, r=r, xi=xi, zeta=zeta, psiR=psiR, psiT=psiT)


def _load_correlators_npz(fname):
    """Save correlation functions.

    Parameters
    ----------
    fname : str
        Filename.

    Returns
    -------
    redshift : float
        Redshift.
    r : array
        Radial axis.
    xi : array
        Density auto-correlation function.
    zeta : array
        Density-velocity cross correlation function.
    PsiR : array
        Velocity auto-correlation radial function.
    PsiT : array
        Velocity auto-correlation tangential function.
    """
    data = np.load(fname)
    redshift = data['redshift']
    r = data['r']
    xi = data['xi']
    zeta = data['zeta']
    psiR = data['psiR']
    psiT = data['psiT']
    return redshift, r, xi, zeta, psiR, psiT


def save_correlators(fname, redshift, r, xi, zeta, psiR, psiT, filetype='npz'):
    """Save correlation functions.

    Parameters
    ----------
    fname : str
        Filename.
    redshift : float
        Redshift.
    r : array
        Radial axis.
    xi : array
        Density auto-correlation function.
    zeta : array
        Density-velocity cross correlation function.
    PsiR : array
        Velocity auto-correlation radial function.
    PsiT : array
        Velocity auto-correlation tangential function.
    filetype : str
        - 'npz' : numpy filetype.
    """
    if filetype == 'npz':
        _save_correlators_npz(fname, redshift, r, xi, zeta, psiR, psiT)


def load_correlators(fname, filetype='npz'):
    """Save correlation functions.

    Parameters
    ----------
    fname : str
        Filename.
    filetype : str
        - 'npz' : numpy filetype.

    Returns
    -------
    redshift : float
        Redshift.
    r : array
        Radial axis.
    xi : array
        Density auto-correlation function.
    zeta : array
        Density-velocity cross correlation function.
    PsiR : array
        Velocity auto-correlation radial function.
    PsiT : array
        Velocity auto-correlation tangential function.
    """
    if filetype == 'npz':
        return _load_correlators_npz(fname)
