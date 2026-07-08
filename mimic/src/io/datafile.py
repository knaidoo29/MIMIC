import numpy as np


def _save_analytic_correlators_npz(fname, redshift, r, xi, zeta_p, zeta_u, psiR_pp, psiT_pp,
    psiR_pu, psiT_pu, psiR_uu, psiT_uu):
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
    zeta_p, zeta_u : array
        Density-velocity cross correlation function.
    PsiR_pp, PsiR_pu, PsiR_uu : array
        Velocity auto-correlation radial function.
    PsiT_pp, PsiT_pu, PsiT_uu : array
        Velocity auto-correlation tangential function.
    """
    np.savez(fname, redshift=redshift, r=r, xi=xi, zeta_p=zeta_p, zeta_u=zeta_u,
        psiR_pp=psiR_pp, psiT_pp=psiT_pp, psiR_pu=psiR_pu, psiT_pu=psiT_pu,
        psiR_uu=psiR_uu, psiT_uu=psiT_uu)


def _load_analytic_correlators_npz(fname):
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
    zeta_p = data['zeta_p']
    zeta_u = data['zeta_u']
    psiR_pp = data['psiR_pp']
    psiT_pp = data['psiT_pp']
    psiR_pu = data['psiR_pu']
    psiT_pu = data['psiT_pu']
    psiR_uu = data['psiR_uu']
    psiT_uu = data['psiT_uu']
    return redshift, r, xi, zeta_p, zeta_u, psiR_pp, psiT_pp, psiR_pu, psiT_pu, psiR_uu, psiT_uu


def save_analytic_correlators(fname, redshift, r, xi, zeta_p, zeta_u, psiR_pp, psiT_pp,
    psiR_pu, psiT_pu, psiR_uu, psiT_uu, filetype='npz'):
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
        _save_analytic_correlators_npz(fname, redshift, r, xi, zeta_p, zeta_u, psiR_pp,
            psiT_pp, psiR_pu, psiT_pu, psiR_uu, psiT_uu)


def load_analytic_correlators(fname, filetype='npz'):
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
        return _load_analytic_correlators_npz(fname)


def _save_grid_correlators_npz(
    fname, redshift, xi, zeta_p, zeta_u, psixx_pp, psixy_pp,
    psixx_pu, psixy_pu, psixx_uu, psixy_uu, stretch
):
    """Save correlation functions.

    Parameters
    ----------
    fname : str
        Filename.
    redshift : float
        Redshift.
    xi : ndarray
        Density auto-correlation function.
    zeta_p, zeta_u : ndarray
        Density-velocity cross correlation function.
    Psixx_pp, Psixx_pu, Psixx_uu : ndarray
        Velocity auto-correlation xx function.
    Psixy_pp, Psixy_pu, Psixy_uu : ndarray
        Velocity auto-correlation xy function.
    """
    np.savez(
        fname, stretch=stretch, redshift=redshift, xi=xi, zeta_p=zeta_p, zeta_u=zeta_u, psixx_pp=psixx_pp, psixy_pp=psixy_pp, 
        psixx_pu=psixx_pu, psixy_pu=psixy_pu, psixx_uu=psixx_uu, psixy_uu=psixy_uu
    )


def _load_grid_correlators_npz(fname):
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
    stretch = data['stretch']
    xi = data['xi']
    zeta_p = data['zeta_p']
    zeta_u = data['zeta_u']
    psixx_pp = data['psixx_pp']
    psixy_pp = data['psixy_pp']
    psixx_pu = data['psixx_pu']
    psixy_pu = data['psixy_pu']
    psixx_uu = data['psixx_uu']
    psixy_uu = data['psixy_uu']
    return redshift, stretch, xi, zeta_p, zeta_u, psixx_pp, psixy_pp, psixx_pu, psixy_pu, psixx_uu, psixy_uu


def save_grid_correlators(fname, redshift, xi, zeta_p, zeta_u, psixx_pp, psixy_pp,
    psixx_pu, psixy_pu, psixx_uu, psixy_uu, stretch, filetype='npz'):
    """Save correlation functions.

    Parameters
    ----------
    fname : str
        Filename.
    redshift : float
        Redshift.
    xi : array
        Density auto-correlation function.
    zeta : array
        Density-velocity cross correlation function.
    Psixx : array
        Velocity auto-correlation xx function.
    Psixx : array
        Velocity auto-correlation xy function.
    stretch : bool
        If true then the grid is sin stretched.
    filetype : str
        - 'npz' : numpy filetype.
    """
    if filetype == 'npz':
        _save_grid_correlators_npz(
            fname, redshift, xi, zeta_p, zeta_u, psixx_pp, psixy_pp, psixx_pu, psixy_pu, psixx_uu, psixy_uu, stretch
        )


def load_grid_correlators(fname, filetype='npz'):
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
        return _load_grid_correlators_npz(fname)