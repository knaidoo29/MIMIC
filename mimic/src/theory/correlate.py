import numpy as np

from . import Hz
from .. import io, src


def _get_adot_phi(redshift):
    """Returns adot for units in Mpc/h.

    Parameters
    ----------
    redshift : float
        Redshift.

    Returns
    -------
    adot : float
        adot in units of Mpc/h.
    """
    a = Hz.z2a(redshift)
    adot = a
    return adot


def _get_adot_vel(redshift, interp_Hz):
    """Returns adot for units in Mpc/h.

    Parameters
    ----------
    redshift : float
        Redshift.
    interp_Hz : func
        Interpolating Hubble function.

    Returns
    -------
    adot : float
        adot in units of km/s.
    """
    a = Hz.z2a(redshift)
    adot = a*interp_Hz(redshift)
    return adot


def get_cc_float_fast(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2, type1, type2, redshift, 
    interp_Hz, interp_xi, interp_zeta_p, interp_zeta_u, interp_psiR_pp, interp_psiT_pp, 
    interp_psiR_pu, interp_psiT_pu, interp_psiR_uu, interp_psiT_uu, boxsize, minlogr=-2, nlogr=1000
):
    """Computes correlation functions.

    Parameters
    ----------
    x1, y1, z1 : float
        Coordinates of point 1.
    x2, y2, z2 : float
        Coordinates of point 2.
    ex1, ey1, ez1 : float
        Directional vector for point 1 constraints or desired constraints.
    ex2, ey2, ez2 : float
        Directional vector for point 2 constraints or desired constraints.
    type1 : int
        Constraint type.
    type2 : int
        Constraint type.
    redshift : float
        Redshift.
    interp_Hz : func
        Expansion rate interpolation function.
    interp_xi : func
        Density autocorrelation function.
    interp_zeta_p, interp_zeta_u : func
        Density to displacement/velocity cross-correlation functions.
    interp_psiR_pp, interp_psiT_pp : func
        Radial and tangential displacment autocorrelation.
    interp_psiR_pu, interp_psiT_pu : func
        Radial and tangential displacment-velocity autocorrelation.
    interp_psiR_uu, interp_psiT_uu : func
        Radial and tangential velocity autocorrelation.
    boxsize : float
        Size of the box.
    minlogr : float, optional
        Minimum value for the fortran correlation function linear interpolator.
    nlogr : int, optional
        Number of divisions in the fortran correlation function linear interpolator.

    Returns
    -------
    cc : float
        Correlation values.
    """

    _r = np.logspace(minlogr, np.log10(np.sqrt(3.)*boxsize), nlogr)
    xi = interp_xi(_r)
    zeta_p = interp_zeta_p(_r)
    zeta_u = interp_zeta_u(_r)
    psiR_pp = interp_psiR_pp(_r)
    psiT_pp = interp_psiT_pp(_r)
    psiR_pu = interp_psiR_pu(_r)
    psiT_pu = interp_psiT_pu(_r)
    psiR_uu = interp_psiR_uu(_r)
    psiT_uu = interp_psiT_uu(_r)

    adot_phi = _get_adot_phi(redshift)
    adot_vel = _get_adot_vel(redshift, interp_Hz)

    cc = src.get_cc_float(
        x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2, type1, type2, adot_phi,
        adot_vel, np.log10(_r), xi, zeta_p, zeta_u, psiR_pp, psiT_pp, psiR_pu, psiT_pu,
        psiR_uu, psiT_uu, boxsize
    )

    return cc


def get_cc_float_fast_grid(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    type1, type2, redshift, interp_Hz, xi_box, zeta_p_box, zeta_u_box, psixx_pp_box, psixy_pp_box, 
    psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box, boxsize, stretch):
    """Computes correlation functions.

    Parameters
    ----------
    x1, y1, z1 : float
        Coordinates of point 1.
    x2, y2, z2 : float
        Coordinates of point 2.
    ex1, ey1, ez1 : float
        Directional vector for point 1 constraints or desired constraints.
    ex2, ey2, ez2 : float
        Directional vector for point 2 constraints or desired constraints.
    type1 : int
        Constraint type.
    type2 : int
        Constraint type.
    redshift : float
        Redshift.
    interp_Hz : func
        Expansion rate interpolation function.
    xi_box, zeta_p_box, zeta_u_box, psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box : arrays
        Correlation profiles on a grid.
    boxsize : float
        Size of the box.
    
    Returns
    -------
    cc : float
        Correlation values.
    """

    adot_phi = _get_adot_phi(redshift)
    adot_vel = _get_adot_vel(redshift, interp_Hz)

    box_shape = np.shape(xi_box)

    if len(box_shape) != 3:
        raise ValueError("Grid correlator box must be a 3D array.")

    if box_shape[0] != box_shape[1] or box_shape[1] != box_shape[2]:
        raise ValueError(
            "Grid correlator box must be the full cubic grid. "
            "Got shape %s. This looks like an MPI slab, not a full grid."
            % (str(box_shape),)
        )

    ngrid = box_shape[0]

    xi_box = xi_box.flatten()
    zeta_p_box = zeta_p_box.flatten()
    zeta_u_box = zeta_u_box.flatten()
    psixx_pp_box = psixx_pp_box.flatten()
    psixy_pp_box = psixy_pp_box.flatten()
    psixx_pu_box = psixx_pu_box.flatten()
    psixy_pu_box = psixy_pu_box.flatten()
    psixx_uu_box = psixx_uu_box.flatten()
    psixy_uu_box = psixy_uu_box.flatten()

    cc = src.get_cc_grid_float(
        x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2, type1, type2, adot_phi,
        adot_vel, xi_box, zeta_p_box, zeta_u_box, psixx_pp_box, psixy_pp_box, 
        psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box, ngrid, boxsize, stretch
    )

    xi_box = xi_box.reshape(box_shape)
    zeta_p_box = zeta_p_box.reshape(box_shape)
    zeta_u_box = zeta_u_box.reshape(box_shape)
    psixx_pp_box = psixx_pp_box.reshape(box_shape)
    psixy_pp_box = psixy_pp_box.reshape(box_shape)
    psixx_pu_box = psixx_pu_box.reshape(box_shape)
    psixy_pu_box = psixy_pu_box.reshape(box_shape)
    psixx_uu_box = psixx_uu_box.reshape(box_shape)
    psixy_uu_box = psixy_uu_box.reshape(box_shape)

    return cc


def get_cc_vector_fast(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2, type1, type2, 
    redshift, interp_Hz, interp_xi, interp_zeta_p, interp_zeta_u, interp_psiR_pp, interp_psiT_pp, 
    interp_psiR_pu, interp_psiT_pu, interp_psiR_uu, interp_psiT_uu, boxsize, minlogr=-2, nlogr=10000
):
    """Computes correlation functions with either point 1 or 2 being an array.

    Parameters
    ----------
    x1, y1, z1 : float/array
        Coordinates of point 1.
    x2, y2, z2 : float/array
        Coordinates of point 2.
    ex1, ey1, ez1 : float/array
        Directional vector for point 1 constraints or desired constraints.
    ex2, ey2, ez2 : float/array
        Directional vector for point 2 constraints or desired constraints.
    type1 : int/array
        Constraint type.
    type2 : int/array
        Constraint type.
    redshift : float
        Redshift.
    interp_Hz : func
        Expansion rate interpolation function.
    interp_xi : func
        Density autocorrelation function.
    interp_zeta_p, interp_zeta_u : func
        Density to displacement/velocity cross-correlation functions.
    interp_psiR_pp, interp_psiT_pp : func
        Radial and tangential displacment autocorrelation.
    interp_psiR_pu, interp_psiT_pu : func
        Radial and tangential displacment-velocity autocorrelation.
    interp_psiR_uu, interp_psiT_uu : func
        Radial and tangential velocity autocorrelation.
    boxsize : float
        Size of the box.
    minlogr : float, optional
        Minimum value for the fortran correlation function linear interpolator.
    nlogr : int, optional
        Number of divisions in the fortran correlation function linear interpolator.

    Returns
    -------
    cc : array
        Correlation values.
    """

    _r = np.logspace(minlogr, np.log10(np.sqrt(3.)*boxsize), nlogr)
    xi = interp_xi(_r)
    zeta_p = interp_zeta_p(_r)
    zeta_u = interp_zeta_u(_r)
    psiR_pp = interp_psiR_pp(_r)
    psiT_pp = interp_psiT_pp(_r)
    psiR_pu = interp_psiR_pu(_r)
    psiT_pu = interp_psiT_pu(_r)
    psiR_uu = interp_psiR_uu(_r)
    psiT_uu = interp_psiT_uu(_r)

    if io.isscalar(x1):
        _shape = np.shape(x2)
        x2, y2, z2 = x2.flatten(), y2.flatten(), z2.flatten()
        ex2, ey2, ez2 = ex2.flatten(), ey2.flatten(), ez2.flatten()
        type2 = type2.flatten()

        adot_phi = _get_adot_phi(redshift)
        adot_vel = _get_adot_vel(redshift, interp_Hz)
        
        cc = src.get_cc_array2(
            x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
            type1, type2, adot_phi, adot_vel, np.log10(_r), xi, zeta_p, zeta_u,
            psiR_pp, psiT_pp, psiR_pu, psiT_pu, psiR_uu, psiT_uu, boxsize
        )

        x2, y2, z2 = x2.reshape(_shape), y2.reshape(_shape), z2.reshape(_shape)
        ex2, ey2, ez2 = ex2.reshape(_shape), ey2.reshape(_shape), ez2.reshape(_shape)
        type2 = type2.reshape(_shape)
        cc = cc.reshape(_shape)
    else:
        _shape = np.shape(x1)
        x1, y1, z1 = x1.flatten(), y1.flatten(), z1.flatten()
        ex1, ey1, ez1 = ex1.flatten(), ey1.flatten(), ez1.flatten()
        type1 = type1.flatten()

        adot_phi = _get_adot_phi(redshift)
        adot_vel = _get_adot_vel(redshift, interp_Hz)
        
        cc = src.get_cc_array1(
            x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
            type1, type2, adot_phi, adot_vel, np.log10(_r), xi, zeta_p, zeta_u,
            psiR_pp, psiT_pp, psiR_pu, psiT_pu, psiR_uu, psiT_uu, boxsize)

        x1, y1, z1 = x1.reshape(_shape), y1.reshape(_shape), z1.reshape(_shape)
        ex1, ey1, ez1 = ex1.reshape(_shape), ey1.reshape(_shape), ez1.reshape(_shape)
        type1 = type1.reshape(_shape)
        cc = cc.reshape(_shape)

    return cc


def get_cc_vector_fast_grid(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2, type1, type2, redshift, 
    interp_Hz, xi_box, zeta_p_box, zeta_u_box, psixx_pp_box, psixy_pp_box, 
    psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box, boxsize, stretch
):
    """Computes correlation functions with either point 1 or 2 being an array.

    Parameters
    ----------
    x1, y1, z1 : float/array
        Coordinates of point 1.
    x2, y2, z2 : float/array
        Coordinates of point 2.
    ex1, ey1, ez1 : float/array
        Directional vector for point 1 constraints or desired constraints.
    ex2, ey2, ez2 : float/array
        Directional vector for point 2 constraints or desired constraints.
    type1 : int/array
        Constraint type.
    type2 : int/array
        Constraint type.
    redshift : float
        Redshift.
    interp_Hz : func
        Expansion rate interpolation function.
    interp_xi : func
        Density autocorrelation function.
    interp_zeta_p, interp_zeta_u : func
        Density to displacement/velocity cross-correlation functions.
    interp_psiR_pp, interp_psiT_pp : func
        Radial and tangential displacment autocorrelation.
    interp_psiR_pu, interp_psiT_pu : func
        Radial and tangential displacment-velocity autocorrelation.
    interp_psiR_uu, interp_psiT_uu : func
        Radial and tangential velocity autocorrelation.
    boxsize : float
        Size of the box.
    minlogr : float, optional
        Minimum value for the fortran correlation function linear interpolator.
    nlogr : int, optional
        Number of divisions in the fortran correlation function linear interpolator.

    Returns
    -------
    cc : array
        Correlation values.
    """

    box_shape = np.shape(xi_box)

    if len(box_shape) != 3:
        raise ValueError("Grid correlator box must be a 3D array.")

    if box_shape[0] != box_shape[1] or box_shape[1] != box_shape[2]:
        raise ValueError(
            "Grid correlator box must be the full cubic grid. "
            "Got shape %s. This looks like an MPI slab, not a full grid."
            % (str(box_shape),)
        )

    ngrid = box_shape[0]

    xi_box = xi_box.flatten()
    zeta_p_box = zeta_p_box.flatten()
    zeta_u_box = zeta_u_box.flatten()
    psixx_pp_box = psixx_pp_box.flatten()
    psixy_pp_box = psixy_pp_box.flatten()
    psixx_pu_box = psixx_pu_box.flatten()
    psixy_pu_box = psixy_pu_box.flatten()
    psixx_uu_box = psixx_uu_box.flatten()
    psixy_uu_box = psixy_uu_box.flatten()

    if io.isscalar(x1):
        _shape = np.shape(x2)
        x2, y2, z2 = x2.flatten(), y2.flatten(), z2.flatten()
        ex2, ey2, ez2 = ex2.flatten(), ey2.flatten(), ez2.flatten()
        type2 = type2.flatten()

        adot_phi = _get_adot_phi(redshift)
        adot_vel = _get_adot_vel(redshift, interp_Hz)

        cc = src.get_cc_grid_array2(
            x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
            type1, type2, adot_phi, adot_vel, xi_box, zeta_p_box, zeta_u_box,
            psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box, 
            ngrid, boxsize, stretch
        )

        x2, y2, z2 = x2.reshape(_shape), y2.reshape(_shape), z2.reshape(_shape)
        ex2, ey2, ez2 = ex2.reshape(_shape), ey2.reshape(_shape), ez2.reshape(_shape)
        type2 = type2.reshape(_shape)
        cc = cc.reshape(_shape)

    else:
        _shape = np.shape(x1)
        x1, y1, z1 = x1.flatten(), y1.flatten(), z1.flatten()
        ex1, ey1, ez1 = ex1.flatten(), ey1.flatten(), ez1.flatten()
        type1 = type1.flatten()

        adot_phi = _get_adot_phi(redshift)
        adot_vel = _get_adot_vel(redshift, interp_Hz)

        cc = src.get_cc_grid_array1(
            x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
            type1, type2, adot_phi, adot_vel, xi_box, zeta_p_box, zeta_u_box,
            psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box, 
            ngrid, boxsize, stretch
        )

        x1, y1, z1 = x1.reshape(_shape), y1.reshape(_shape), z1.reshape(_shape)
        ex1, ey1, ez1 = ex1.reshape(_shape), ey1.reshape(_shape), ez1.reshape(_shape)
        type1 = type1.reshape(_shape)
        cc = cc.reshape(_shape)

    xi_box = xi_box.reshape(box_shape)
    zeta_p_box = zeta_p_box.reshape(box_shape)
    zeta_u_box = zeta_u_box.reshape(box_shape)
    psixx_pp_box = psixx_pp_box.reshape(box_shape)
    psixy_pp_box = psixy_pp_box.reshape(box_shape)
    psixx_pu_box = psixx_pu_box.reshape(box_shape)
    psixy_pu_box = psixy_pu_box.reshape(box_shape)
    psixx_uu_box = psixx_uu_box.reshape(box_shape)
    psixy_uu_box = psixy_uu_box.reshape(box_shape)

    return cc


def get_cc_matrix_fast(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2, type1, type2, redshift, 
    interp_Hz, interp_xi, interp_zeta_p, interp_zeta_u, interp_psiR_pp, interp_psiT_pp, 
    interp_psiR_pu, interp_psiT_pu, interp_psiR_uu, interp_psiT_uu, boxsize, minlogr=-2, 
    nlogr=1000
):
    """Computes correlation functions with either point 1 or 2 being an array.

    Parameters
    ----------
    x1, y1, z1 : array
        Coordinates of point 1.
    x2, y2, z2 : array
        Coordinates of point 2.
    ex1, ey1, ez1 : array
        Directional vector for point 1 constraints or desired constraints.
    ex2, ey2, ez2 : array
        Directional vector for point 2 constraints or desired constraints.
    type1 : array
        Constraint type.
    type2 : array
        Constraint type.
    redshift : float
        Redshift.
    interp_Hz : func
        Expansion rate interpolation function.
    interp_xi : func
        Density autocorrelation function.
    interp_zeta_p, interp_zeta_u : func
        Density to displacement/velocity cross-correlation functions.
    interp_psiR_pp, interp_psiT_pp : func
        Radial and tangential displacment autocorrelation.
    interp_psiR_pu, interp_psiT_pu : func
        Radial and tangential displacment-velocity autocorrelation.
    interp_psiR_uu, interp_psiT_uu : func
        Radial and tangential velocity autocorrelation.
    boxsize : float
        Size of the box.
    minlogr : float, optional
        Minimum value for the fortran correlation function linear interpolator.
    nlogr : int, optional
        Number of divisions in the fortran correlation function linear interpolator.

    Returns
    -------
    cc : array
        Correlation values.
    """

    _r = np.logspace(minlogr, np.log10(np.sqrt(3.)*boxsize), nlogr)
    xi = interp_xi(_r)
    zeta_p = interp_zeta_p(_r)
    zeta_u = interp_zeta_u(_r)
    psiR_pp = interp_psiR_pp(_r)
    psiT_pp = interp_psiT_pp(_r)
    psiR_pu = interp_psiR_pu(_r)
    psiT_pu = interp_psiT_pu(_r)
    psiR_uu = interp_psiR_uu(_r)
    psiT_uu = interp_psiT_uu(_r)

    _shape = np.shape(x1)
    x1, y1, z1 = x1.flatten(), y1.flatten(), z1.flatten()
    x2, y2, z2 = x2.flatten(), y2.flatten(), z2.flatten()
    ex1, ey1, ez1 = ex1.flatten(), ey1.flatten(), ez1.flatten()
    ex2, ey2, ez2 = ex2.flatten(), ey2.flatten(), ez2.flatten()
    type1, type2 = type1.flatten(), type2.flatten()

    adot_phi = _get_adot_phi(redshift)
    adot_vel = _get_adot_vel(redshift, interp_Hz)

    cc = src.get_cc_arrays(
        x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
        type1, type2, adot_phi, adot_vel, np.log10(_r), xi, zeta_p, zeta_u,
        psiR_pp, psiT_pp, psiR_pu, psiT_pu, psiR_uu, psiT_uu, boxsize
    )

    x1, y1, z1 = x1.reshape(_shape), y1.reshape(_shape), z1.reshape(_shape)
    x2, y2, z2 = x2.reshape(_shape), y2.reshape(_shape), z2.reshape(_shape)
    ex1, ey1, ez1 = ex1.reshape(_shape), ey1.reshape(_shape), ez1.reshape(_shape)
    ex2, ey2, ez2 = ex2.reshape(_shape), ey2.reshape(_shape), ez2.reshape(_shape)
    type1, type2 = type1.reshape(_shape), type2.reshape(_shape)
    cc = cc.reshape(_shape)

    return cc


def get_cc_matrix_fast_grid(
    x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2, type1, type2, 
    redshift, interp_Hz, xi_box, zeta_p_box, zeta_u_box, psixx_pp_box, psixy_pp_box, 
    psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box, boxsize, stretch
):
    """Computes correlation functions with either point 1 or 2 being an array.

    Parameters
    ----------
    x1, y1, z1 : array
        Coordinates of point 1.
    x2, y2, z2 : array
        Coordinates of point 2.
    ex1, ey1, ez1 : array
        Directional vector for point 1 constraints or desired constraints.
    ex2, ey2, ez2 : array
        Directional vector for point 2 constraints or desired constraints.
    type1 : array
        Constraint type.
    type2 : array
        Constraint type.
    redshift : float
        Redshift.
    interp_Hz : func
        Expansion rate interpolation function.
    boxsize : float
        Size of the box.
    minlogr : float, optional
        Minimum value for the fortran correlation function linear interpolator.
    nlogr : int, optional
        Number of divisions in the fortran correlation function linear interpolator.

    Returns
    -------
    cc : array
        Correlation values.
    """

    box_shape = np.shape(xi_box)

    if len(box_shape) != 3:
        raise ValueError("Grid correlator box must be a 3D array.")

    if box_shape[0] != box_shape[1] or box_shape[1] != box_shape[2]:
        raise ValueError(
            "Grid correlator box must be the full cubic grid. "
            "Got shape %s. This looks like an MPI slab, not a full grid."
            % (str(box_shape),)
        )

    ngrid = box_shape[0]

    xi_box = xi_box.flatten()
    zeta_p_box = zeta_p_box.flatten()
    zeta_u_box = zeta_u_box.flatten()
    psixx_pp_box = psixx_pp_box.flatten()
    psixy_pp_box = psixy_pp_box.flatten()
    psixx_pu_box = psixx_pu_box.flatten()
    psixy_pu_box = psixy_pu_box.flatten()
    psixx_uu_box = psixx_uu_box.flatten()
    psixy_uu_box = psixy_uu_box.flatten()

    _shape = np.shape(x1)
    x1, y1, z1 = x1.flatten(), y1.flatten(), z1.flatten()
    x2, y2, z2 = x2.flatten(), y2.flatten(), z2.flatten()
    ex1, ey1, ez1 = ex1.flatten(), ey1.flatten(), ez1.flatten()
    ex2, ey2, ez2 = ex2.flatten(), ey2.flatten(), ez2.flatten()
    type1, type2 = type1.flatten(), type2.flatten()

    adot_phi = _get_adot_phi(redshift)
    adot_vel = _get_adot_vel(redshift, interp_Hz)

    cc = src.get_cc_grid_arrays(
        x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
        type1, type2, adot_phi, adot_vel, xi_box, zeta_p_box, zeta_u_box,
        psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box, psixx_uu_box, psixy_uu_box, 
        ngrid, boxsize, stretch
    )

    x1, y1, z1 = x1.reshape(_shape), y1.reshape(_shape), z1.reshape(_shape)
    x2, y2, z2 = x2.reshape(_shape), y2.reshape(_shape), z2.reshape(_shape)
    ex1, ey1, ez1 = ex1.reshape(_shape), ey1.reshape(_shape), ez1.reshape(_shape)
    ex2, ey2, ez2 = ex2.reshape(_shape), ey2.reshape(_shape), ez2.reshape(_shape)
    type1, type2 = type1.reshape(_shape), type2.reshape(_shape)
    cc = cc.reshape(_shape)

    xi_box = xi_box.reshape(box_shape)
    zeta_p_box = zeta_p_box.reshape(box_shape)
    zeta_u_box = zeta_u_box.reshape(box_shape)
    psixx_pp_box = psixx_pp_box.reshape(box_shape)
    psixy_pp_box = psixy_pp_box.reshape(box_shape)
    psixx_pu_box = psixx_pu_box.reshape(box_shape)
    psixy_pu_box = psixy_pu_box.reshape(box_shape)
    psixx_uu_box = psixx_uu_box.reshape(box_shape)
    psixy_uu_box = psixy_uu_box.reshape(box_shape)

    return cc


def get_corr_dot_eta_fast(
    x1, xc, y1, yc, z1, zc, ex1, exc, ey1, eyc, ez1, ezc, type1, typec, redshift, 
    interp_Hz, interp_xi, interp_zeta_p, interp_zeta_u, interp_psiR_pp, interp_psiT_pp, 
    interp_psiR_pu, interp_psiT_pu, interp_psiR_uu, interp_psiT_uu, eta, boxsize, 
    lenpro, prefix, mpi_rank=0, minlogr=-2, nlogr=1000
):
    """Computes the dot product of a correlation function and eta, inverse
    covariance dot constraint values.

    Parameters
    ----------
    x1, y1, z1 : array
        Coordinates of point 1.
    xc, yc, zc : array
        Coordinates of point c.
    ex1, ey1, ez1 : float/array
        Directional vector for point 1 constraints or desired constraints.
    exc, eyc, ezc : array
        Directional vector for point c constraints or desired constraints.
    type1 : array
        Constraint type.
    typec : array
        Constraint type.
    redshift : float
        Redshift.
    interp_Hz : func
        Expansion rate interpolation function.
    interp_xi : func
        Density autocorrelation function.
    interp_zeta_p, interp_zeta_u : func
        Density to displacement/velocity cross-correlation functions.
    interp_psiR_pp, interp_psiT_pp : func
        Radial and tangential displacment autocorrelation.
    interp_psiR_pu, interp_psiT_pu : func
        Radial and tangential displacment-velocity autocorrelation.
    interp_psiR_uu, interp_psiT_uu : func
        Radial and tangential velocity autocorrelation.
    eta : array
        Inverse covariance dot constraint values.
    boxsize : float
        Size of the box.
    lenpro : int
        Length of the progress bar.
    prefix : str
        Prefix for fortran progress bar.
    mpi_rank : int, optional
        The rank of the MPI object.
    minlogr : float, optional
        Minimum value for the fortran correlation function linear interpolator.
    nlogr : int, optional
        Number of divisions in the fortran correlation function linear interpolator.

    Returns
    -------
    field : array
        Results of the dot product.
    """

    _r = np.logspace(minlogr, np.log10(np.sqrt(3.)*boxsize), nlogr)
    xi = interp_xi(_r)
    zeta_p = interp_zeta_p(_r)
    zeta_u = interp_zeta_u(_r)
    psiR_pp = interp_psiR_pp(_r)
    psiT_pp = interp_psiT_pp(_r)
    psiR_pu = interp_psiR_pu(_r)
    psiT_pu = interp_psiT_pu(_r)
    psiR_uu = interp_psiR_uu(_r)
    psiT_uu = interp_psiT_uu(_r)

    _shape1 = np.shape(x1)
    _shapec = np.shape(xc)
    x1, y1, z1 = x1.flatten(), y1.flatten(), z1.flatten()
    xc, yc, zc = xc.flatten(), yc.flatten(), zc.flatten()
    if io.isscalar(ex1) is not True:
        ex1, ey1, ez1 = ex1.flatten(), ey1.flatten(), ez1.flatten()
    exc, eyc, ezc = exc.flatten(), eyc.flatten(), ezc.flatten()
    # type1 = type1.flatten()
    typec = typec.flatten()

    adot_phi = _get_adot_phi(redshift)
    adot_vel = _get_adot_vel(redshift, interp_Hz)

    if io.isscalar(ex1):
        field = src.corr_dot_eta(
            x1, xc, y1, yc, z1, zc, ex1, exc, ey1, eyc, ez1, ezc,
            type1, typec, adot_phi, adot_vel, np.log10(_r), xi, zeta_p, zeta_u,
            psiR_pp, psiT_pp, psiR_pu, psiT_pu, psiR_uu, psiT_uu,
            boxsize, eta, mpi_rank=mpi_rank, lenpro=lenpro, prefix=prefix
        )
    else:
        field = src.corr_dot_eta_array(
            x1, xc, y1, yc, z1, zc, ex1, exc, ey1, eyc, ez1, ezc,
            type1, typec, adot_phi, adot_vel, np.log10(_r), xi, zeta_p, zeta_u,
            psiR_pp, psiT_pp, psiR_pu, psiT_pu, psiR_uu, psiT_uu,
            boxsize, eta, mpi_rank=mpi_rank, lenpro=lenpro, prefix=prefix
        )

    x1, y1, z1 = x1.reshape(_shape1), y1.reshape(_shape1), z1.reshape(_shape1)
    xc, yc, zc = xc.reshape(_shapec), yc.reshape(_shapec), zc.reshape(_shapec)
    if io.isscalar(ex1) is not True:
        ex1, ey1, ez1 = ex1.reshape(_shape1), ey1.reshape(_shape1), ez1.reshape(_shape1)
        
    exc, eyc, ezc = exc.reshape(_shapec), eyc.reshape(_shapec), ezc.reshape(_shapec)
    # type1 = type1.reshape(_shape1)
    typec = typec.reshape(_shapec)
    field = field.reshape(_shape1)

    return field


def get_corr_dot_eta_fast_grid(
    x1, xc, y1, yc, z1, zc, ex1, exc, ey1, eyc, ez1, ezc, type1, typec, redshift, interp_Hz, 
    xi_box, zeta_p_box, zeta_u_box, psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box,
    psixx_uu_box, psixy_uu_box, eta, boxsize, lenpro, prefix, stretch, mpi_rank=0
):
    """Computes the dot product of a correlation function and eta, inverse
    covariance dot constraint values.

    Parameters
    ----------
    x1, y1, z1 : array
        Coordinates of point 1.
    xc, yc, zc : array
        Coordinates of point c.
    ex1, ey1, ez1 : float/array
        Directional vector for point 1 constraints or desired constraints.
    exc, eyc, ezc : array
        Directional vector for point c constraints or desired constraints.
    type1 : array
        Constraint type.
    typec : array
        Constraint type.
    redshift : float
        Redshift.
    interp_Hz : func
        Expansion rate interpolation function.
    interp_xi : func
        Density autocorrelation function.
    interp_zeta_p, interp_zeta_u : func
        Density to displacement/velocity cross-correlation functions.
    interp_psiR_pp, interp_psiT_pp : func
        Radial and tangential displacment autocorrelation.
    interp_psiR_pu, interp_psiT_pu : func
        Radial and tangential displacment-velocity autocorrelation.
    interp_psiR_uu, interp_psiT_uu : func
        Radial and tangential velocity autocorrelation.
    eta : array
        Inverse covariance dot constraint values.
    boxsize : float
        Size of the box.
    lenpro : int
        Length of the progress bar.
    prefix : str
        Prefix for fortran progress bar.
    mpi_rank : int, optional
        The rank of the MPI object.
    minlogr : float, optional
        Minimum value for the fortran correlation function linear interpolator.
    nlogr : int, optional
        Number of divisions in the fortran correlation function linear interpolator.

    Returns
    -------
    field : array
        Results of the dot product.
    """

    box_shape = np.shape(xi_box)

    if len(box_shape) != 3:
        raise ValueError("Grid correlator box must be a 3D array.")

    if box_shape[0] != box_shape[1] or box_shape[1] != box_shape[2]:
        raise ValueError(
            "Grid correlator box must be the full cubic grid. "
            "Got shape %s. This looks like an MPI slab, not a full grid."
            % (str(box_shape),)
        )

    ngrid = box_shape[0]

    xi_box = xi_box.flatten()
    zeta_p_box = zeta_p_box.flatten()
    zeta_u_box = zeta_u_box.flatten()
    psixx_pp_box = psixx_pp_box.flatten()
    psixy_pp_box = psixy_pp_box.flatten()
    psixx_pu_box = psixx_pu_box.flatten()
    psixy_pu_box = psixy_pu_box.flatten()
    psixx_uu_box = psixx_uu_box.flatten()
    psixy_uu_box = psixy_uu_box.flatten()

    _shape1 = np.shape(x1)
    _shapec = np.shape(xc)
    x1, y1, z1 = x1.flatten(), y1.flatten(), z1.flatten()
    xc, yc, zc = xc.flatten(), yc.flatten(), zc.flatten()
    if io.isscalar(ex1) is not True:
        ex1, ey1, ez1 = ex1.flatten(), ey1.flatten(), ez1.flatten()
    exc, eyc, ezc = exc.flatten(), eyc.flatten(), ezc.flatten()
    typec = typec.flatten()
    # type1 = type1.flatten()

    adot_phi = _get_adot_phi(redshift)
    adot_vel = _get_adot_vel(redshift, interp_Hz)

    if io.isscalar(ex1):
        field = src.corr_dot_eta_grid(
            x1, xc, y1, yc, z1, zc, ex1, exc, ey1, eyc, ez1, ezc, type1, typec, adot_phi, adot_vel,
            xi_box, zeta_p_box, zeta_u_box, psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box,
            psixx_uu_box, psixy_uu_box, ngrid, boxsize, eta, stretch, mpi_rank=mpi_rank, lenpro=lenpro, prefix=prefix
        )
    else:
        field = src.corr_dot_eta_array_grid(
            x1, xc, y1, yc, z1, zc, ex1, exc, ey1, eyc, ez1, ezc, type1, typec, adot_phi, adot_vel, 
            xi_box, zeta_p_box, zeta_u_box, psixx_pp_box, psixy_pp_box, psixx_pu_box, psixy_pu_box,
            psixx_uu_box, psixy_uu_box, ngrid, boxsize, eta, stretch, mpi_rank=mpi_rank, lenpro=lenpro, prefix=prefix
        )

    x1, y1, z1 = x1.reshape(_shape1), y1.reshape(_shape1), z1.reshape(_shape1)
    xc, yc, zc = xc.reshape(_shapec), yc.reshape(_shapec), zc.reshape(_shapec)
    if io.isscalar(ex1) is not True:
        ex1, ey1, ez1 = ex1.reshape(_shape1), ey1.reshape(_shape1), ez1.reshape(_shape1)
        
    exc, eyc, ezc = exc.reshape(_shapec), eyc.reshape(_shapec), ezc.reshape(_shapec)
    typec = typec.reshape(_shapec)
    # type1 = type1.reshape(_shape1)
    field = field.reshape(_shape1)

    xi_box = xi_box.reshape(box_shape)
    zeta_p_box = zeta_p_box.reshape(box_shape)
    zeta_u_box = zeta_u_box.reshape(box_shape)
    psixx_pp_box = psixx_pp_box.reshape(box_shape)
    psixy_pp_box = psixy_pp_box.reshape(box_shape)
    psixx_pu_box = psixx_pu_box.reshape(box_shape)
    psixy_pu_box = psixy_pu_box.reshape(box_shape)
    psixx_uu_box = psixx_uu_box.reshape(box_shape)
    psixy_uu_box = psixy_uu_box.reshape(box_shape)

    return field