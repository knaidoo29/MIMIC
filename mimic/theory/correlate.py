import numpy as np


from . import coords, Hz
from ..io import isscalar, progress_bar
from .. import src


def _get_adot_phi(redshift):
    a = Hz.z2a(redshift)
    adot = a
    return adot


def _get_adot_vel(redshift, interp_Hz):
    a = Hz.z2a(redshift)
    adot = a*interp_Hz(redshift)
    return adot


def get_cc_vector_fast(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    type1, type2, redshift, interp_Hz, interp_xi, interp_zeta_p, interp_zeta_u,
    interp_psiR_pp, interp_psiT_pp, interp_psiR_pu, interp_psiT_pu, interp_psiR_uu,
    interp_psiT_uu, boxsize, minlogr=-2):

    _r = np.logspace(minlogr, np.log10(np.sqrt(3.)*boxsize), 1000)
    xi = interp_xi(_r)
    zeta_p = interp_zeta_p(_r)
    zeta_u = interp_zeta_u(_r)
    psiR_pp = interp_psiR_pp(_r)
    psiT_pp = interp_psiT_pp(_r)
    psiR_pu = interp_psiR_pu(_r)
    psiT_pu = interp_psiT_pu(_r)
    psiR_uu = interp_psiR_uu(_r)
    psiT_uu = interp_psiT_uu(_r)

    if isscalar(x1):
        _shape = np.shape(x2)
        x2, y2, z2 = x2.flatten(), y2.flatten(), z2.flatten()
        ex2, ey2, ez2 = ex2.flatten(), ey2.flatten(), ez2.flatten()
        type2 = type2.flatten()

        adot_phi = _get_adot_phi(redshift)
        adot_vel = _get_adot_vel(redshift, interp_Hz)

        cc = src.get_cc_array2(x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2, ex1=ex1, ex2=ex2,
            ey1=ey1, ey2=ey2, ez1=ez1, ez2=ez2, type1=type1, type2=type2, adot_phi=adot_phi,
            adot_vel=adot_vel, logr=np.log10(_r), xi=xi, zeta_p=zeta_p, zeta_u=zeta_u, psir_pp=psiR_pp,
            psit_pp=psiT_pp, psir_pu=psiR_pu, psit_pu=psiT_pu, psir_uu=psiR_uu, psit_uu=psiT_uu,
            boxsize=boxsize, lenr=len(_r), lenx2=len(x2))

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

        cc = src.get_cc_array1(x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2, ex1=ex1, ex2=ex2,
            ey1=ey1, ey2=ey2, ez1=ez1, ez2=ez2, type1=type1, type2=type2, adot_phi=adot_phi,
            adot_vel=adot_vel, logr=np.log10(_r), xi=xi, zeta_p=zeta_p, zeta_u=zeta_u, psir_pp=psiR_pp,
            psit_pp=psiT_pp, psir_pu=psiR_pu, psit_pu=psiT_pu, psir_uu=psiR_uu, psit_uu=psiT_uu,
            boxsize=boxsize, lenr=len(_r), lenx1=len(x1))

        x1, y1, z1 = x1.reshape(_shape), y1.reshape(_shape), z1.reshape(_shape)
        ex1, ey1, ez1 = ex1.reshape(_shape), ey1.reshape(_shape), ez1.reshape(_shape)
        type1 = type1.reshape(_shape)
        cc = cc.reshape(_shape)

    return cc


def get_cc_matrix_fast(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    type1, type2, redshift, interp_Hz, interp_xi, interp_zeta_p, interp_zeta_u,
    interp_psiR_pp, interp_psiT_pp, interp_psiR_pu, interp_psiT_pu, interp_psiR_uu,
    interp_psiT_uu, boxsize, minlogr=-2):

    _r = np.logspace(minlogr, np.log10(np.sqrt(3.)*boxsize), 1000)
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

    cc = src.get_cc_arrays(x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2, ex1=ex1, ex2=ex2,
        ey1=ey1, ey2=ey2, ez1=ez1, ez2=ez2, type1=type1, type2=type2, adot_phi=adot_phi,
        adot_vel=adot_vel, logr=np.log10(_r), xi=xi, zeta_p=zeta_p, zeta_u=zeta_u, psir_pp=psiR_pp,
        psit_pp=psiT_pp, psir_pu=psiR_pu, psit_pu=psiT_pu, psir_uu=psiR_uu, psit_uu=psiT_uu,
        boxsize=boxsize, lenr=len(_r), lenx=len(x1))

    x1, y1, z1 = x1.reshape(_shape), y1.reshape(_shape), z1.reshape(_shape)
    x2, y2, z2 = x2.reshape(_shape), y2.reshape(_shape), z2.reshape(_shape)
    ex1, ey1, ez1 = ex1.reshape(_shape), ey1.reshape(_shape), ez1.reshape(_shape)
    ex2, ey2, ez2 = ex2.reshape(_shape), ey2.reshape(_shape), ez2.reshape(_shape)
    type1, type2 = type1.reshape(_shape), type2.reshape(_shape)
    cc = cc.reshape(_shape)

    return cc


def get_cc_dot_eta_fast(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
    type1, type2, redshift, interp_Hz, interp_xi, interp_zeta_p, interp_zeta_u,
    interp_psiR_pp, interp_psiT_pp, interp_psiR_pu, interp_psiT_pu, interp_psiR_uu,
    interp_psiT_uu, eta, boxsize, lenpro, lenpre, prefix, mpi_rank=0, minlogr=-2):

    _r = np.logspace(minlogr, np.log10(np.sqrt(3.)*boxsize), 1000)
    xi = interp_xi(_r)
    zeta_p = interp_zeta_p(_r)
    zeta_u = interp_zeta_u(_r)
    psiR_pp = interp_psiR_pp(_r)
    psiT_pp = interp_psiT_pp(_r)
    psiR_pu = interp_psiR_pu(_r)
    psiT_pu = interp_psiT_pu(_r)
    psiR_uu = interp_psiR_uu(_r)
    psiT_uu = interp_psiT_uu(_r)

    _1shape = np.shape(x1)
    _2shape = np.shape(x2)
    x1, y1, z1 = x1.flatten(), y1.flatten(), z1.flatten()
    x2, y2, z2 = x2.flatten(), y2.flatten(), z2.flatten()
    ex2, ey2, ez2 = ex2.flatten(), ey2.flatten(), ez2.flatten()
    type2 = type2.flatten()

    adot_phi = _get_adot_phi(redshift)
    adot_vel = _get_adot_vel(redshift, interp_Hz)

    field = src.corr_dot_eta(x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2, ex1=ex1, ex2=ex2,
        ey1=ey1, ey2=ey2, ez1=ez1, ez2=ez2, type1=type1, type2=type2, adot_phi=adot_phi,
        adot_vel=adot_vel, logr=np.log10(_r), xi=xi, zeta_p=zeta_p, zeta_u=zeta_u, psir_pp=psiR_pp,
        psit_pp=psiT_pp, psir_pu=psiR_pu, psit_pu=psiT_pu, psir_uu=psiR_uu, psit_uu=psiT_uu,
        boxsize=boxsize, lenr=len(_r), lenx1=len(x1), lenx2=len(x2), eta=eta, mpi_rank=mpi_rank,
        lenpro=lenpro, lenpre=lenpre, prefix=prefix)

    x1, y1, z1 = x1.reshape(_1shape), y1.reshape(_1shape), z1.reshape(_1shape)
    x2, y2, z2 = x2.reshape(_2shape), y2.reshape(_2shape), z2.reshape(_2shape)
    ex2, ey2, ez2 = ex2.reshape(_2shape), ey2.reshape(_2shape), ez2.reshape(_2shape)
    type2 = type2.reshape(_2shape)
    field = field.reshape(_1shape)

    return field
