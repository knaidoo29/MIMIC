import numpy as np

from .. import utils


def write_scale_ind_growth(fname_prefix, redshifts, Dz, fz, ftype='ascii', writelevel=2,
                           verbose=True):
    """Write scale-independent growth functions.

    Parameters
    ----------
    fname_prefix : str
        Scale-independent growth functions prefix for filename.
    redshifts : array
        Redshift array.
    Dz : array
        Growth function.
    fz : array
        Growth rate.
    ftype : str, optional
        File type output.
    writelevel : int, optional
        Print indentation level.
    verbose : bool, optional
        Whether to write or not.
    """
    if ftype == 'ascii':
        # write scale-independent growth functions
        utils.printout('Writing ascii file: '+fname_prefix+'_growth_scale_ind.txt', writelevel, verbose)
        np.savetxt(fname_prefix + '_growth_scale_ind.txt', np.dstack([redshift, Dz, fz])[0], header="z\t D(z)\t f(z)", fmt="%.6f")
    elif ftype == 'numpy':
        # write scale-independent growth functions
        utils.printout('Writing numpy file: '+fname_prefix+'_growth_scale_ind.npy', writelevel, verbose)
        np.savez(fname_prefix + '_growth_scale_ind.npz', redshifts=redshifts, Dz=Dz, fz=fz)


def write_scale_dep_growth(fname_prefix, redshifts, kh, Dzk, fzk, ftype='ascii',
                           writelevel=2, verbose=True):
    """Write scale-dependent growth functions.

    Parameters
    ----------
    fname_prefix : str
        Scale-dependent growth functions.
    redshifts : array
        Redshift array.
    kh : array
        Fourier-modes array.
    Dzk : 2darray
        Growth function.
    fzk : 2darray
        Growth rate.
    ftype : str, optional
        File type output.
    writelevel : int, optional
        Print indentation level.
    verbose : bool, optional
        Whether to write or not.
    """
    if ftype == 'ascii':
        # write scale-dependent redshift values
        utils.printout('Writing ascii file: '+fname_prefix+'_growth_scale_dep_z.txt', writelevel, verbose)
        np.savetxt(fname_prefix + '_growth_scale_dep_z.txt', redshifts, fmt="%.6f")
        # write scale-dependent Fourier modes
        utils.printout('Writing ascii file: '+fname_prefix+'_growth_scale_dep_k.txt', writelevel, verbose)
        np.savetxt(fname_prefix + '_growth_scale_dep_k.txt', kh, fmt="%.6f")
        # write scale-dependent growth function
        utils.printout('Writing ascii file: '+fname_prefix+'_growth_scale_dep_Dzk.txt', writelevel, verbose)
        np.savetxt(fname_prefix + '_growth_scale_dep_D.txt', Dzk, fmt="%.6f")
        # write scale-dependent growth rate
        utils.printout('Writing ascii file: '+fname_prefix+'_growth_scale_dep_fzk.txt', writelevel, verbose)
        np.savetxt(fname_prefix + '_growth_scale_dep_f.txt', fzk, fmt="%.6f")
    elif ftype == 'numpy':
        # write scale-dependent growth functions
        utils.printout('Writing numpy file: '+fname_prefix+'_growth_scale_dep.npy', writelevel, verbose)
        np.savez(fname_prefix + '_growth_scale_dep.npz', redshifts=redshifts, kh=kh, Dzk=Dzk, fzk=fzk)
