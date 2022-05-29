import numpy as np


def read_redshift(zfname, verbose=True):
    """Read redshift file to get redshift values.
    
    Parameters
    ----------
    zfname : str
        Redshift filename.

    Returns
    -------
    redshift : array
        Redshift values.
    """
    redshift = np.loadtxt(zfname, unpack=True)
    return redshift


def read_pk(pkfname, verbose=True, khcol=0, pkcol=1):
    """Read power spectra file.

    Parameters
    ----------
    pkfname : str
        Power spectra filename.
    khcol : int
        Index for the kh column.
    pkcol : int
        Index for the pk column.

    Returns
    -------
    kh, pk : array
        Fourier modes (kh) and Power spectra (pk) arrays.
    """
    data = np.loadtxt(pkfname, unpack=True)
    kh, pk = data[khcol], data[pkcol]
    return kh, pk


def read_pks(pkprefix, zfname, pksuffix='.dat', dp=4, verbose=True, **pkkwargs):
    """Read power spectra files.

    Parameters
    ----------
    pkprefix : str
        Power spectra filename prefix.
    zfname : str
        Redshift filename.
    pksuffix : str
        Power spectra filename suffix after redshift.
    dp : int, optional
        Decimal places for redshift string in power spectra filenames.
    pkkwargs : optional
        read_pk function kwargs.

    Returns
    -------
    redshift : array
        Redshift values.
    kh : array
        Fourier modes.
    pks : array
        Power spectra arrays.
    """
    redshift = read_redshift(zfname)
    if dp is not None:
        redshift = np.round(redshift, decimals=dp)
    redshift_str = np.copy(redshift).astype('str')
    pks = []
    for i in range(0, len(redshift)):
        pkfname = pk_prefix + redshift_str[i] + pk_suffix
        kh, pk = read_pk(pkfname, **pkkwargs)
        pks.append(pk)
    pks = np.array(pks)
    return redshift, kh, pks
