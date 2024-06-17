import numpy as np
from scipy.stats import chi2 as sc_chi2


def get_chi2(d, cov, inv=None):
    """Returns the chi squared from a given vector and covariance.

    Parameters
    ----------
    d : array
        data array.
    cov : matrix
        Covariance matrix.
    inv : matrix, optional
        Inverse of the convariance, provide if the covariance is not changing.
    """
    if inv is None:
        inv = np.linalg.inv(cov)
    return d.dot(inv.dot(d))


def add_sigma_NL(cov, sigma_NL):
    """Add an additional scalar dispersion to the diagonal of a covariance matrix.

    Parameters
    ----------
    cov : matrix
        Covariance matrix.
    sigma_NL : float
        Scalar dispersion value.

    Returns
    -------
    newcov : matrix
        New covariance with added dispersion along the diagonal.
    """
    newcov = cov + np.diag(np.ones(len(cov))*sigma_NL**2.)
    return newcov


def optimize_sigma_NL(c, cov, max_sig_NL=500., etol=0.01, prefix='',
    verbose=True, MPI=None):
    """Optimises the sigma dispersion value to give a covariance function with
    Gaussian errors.

    Parameters
    ----------
    c : array
        Constraint vector.
    cov : matrix
        Covariance with theoretical and measurement errors contributions.
    max_sig_NL : float
        The starting maximum value for the sigma_NL.
    etol : float
        Tolerance in the CDF of the chi2 distribution. The function will look for
        a positive sigma_NL that gives a chi2 CDF of 0.5 +/- etol.
    prefix : str, optional
        For progression statements.
    verbose: bool, optional
        Whether to write outputs.
    MPI : object, optional
        If running on an MPI job, mainly to keep print statements to rank zero.

    Returns
    -------
    success : bool
        True if optimisation was successful.
    sigma_NL : float
        Optimised sigma_NL if success if True.
    """
    df = len(c)
    min_sig_NL=0.
    min_cov = add_sigma_NL(cov, min_sig_NL)
    min_chi2 = get_chi2(c, min_cov)
    max_cov = add_sigma_NL(cov, max_sig_NL)
    max_chi2 = get_chi2(c, max_cov)
    min_cdf = sc_chi2.cdf(min_chi2, df)
    if min_cdf < 0.5:
        if min_cdf < 0.5 - 0.95/2.:
            if verbose:
                if MPI is None:
                    print(prefix+"No sigma_NL required")
                    print(prefix+"chi2 = %0.2f, CDF = %0.2f" % (min_chi2, min_cdf))
                else:
                    MPI.mpi_print_zero(prefix+"No sigma_NL required")
                    MPI.mpi_print_zero(prefix+"chi2 = %0.2f, CDF = %0.2f" % (min_chi2, min_cdf))
            success = True
            sigma_NL = min_sig_NL
        else:
            if verbose:
                if MPI is None:
                    print(prefix+"chi2 is too low for Gaussian field, errors appear to be overestimated.")
                    print(prefix+"chi2 = %0.2f, CDF = %0.2f" % (min_chi2, min_cdf))
                else:
                    MPI.mpi_print_zero(prefix+"chi2 is too low for Gaussian field, errors appear to be overestimated.")
                    MPI.mpi_print_zero(prefix+"chi2 = %0.2f, CDF = %0.2f" % (min_chi2, min_cdf))
            success = False
            sigma_NL = min_sig_NL
    else:
        max_cdf = sc_chi2.cdf(max_chi2, df)
        if verbose:
            if MPI is None:
                print(prefix+"Checking range")
            else:
                MPI.mpi_print_zero(prefix+"Checking range")
        while max_cdf > 0.5:
            min_sig_NL = np.copy(max_sig_NL)
            max_sig_NL = 1.1*max_sig_NL
            max_cov = add_sigma_NL(cov, max_sig_NL)
            max_chi2 = get_chi2(c, max_cov)
            max_cdf = sc_chi2.cdf(max_chi2, df)
            if verbose:
                if MPI is None:
                    print(prefix+"- sigma_NL = %0.2f, chi2 = %0.2f, CDF = %0.2f" % (max_sig_NL, max_chi2, max_cdf))
                else:
                    MPI.mpi_print_zero(prefix+"- sigma_NL = %0.2f, chi2 = %0.2f, CDF = %0.2f" % (max_sig_NL, max_chi2, max_cdf))
        half_sig_NL = (max_sig_NL + min_sig_NL)/2.
        half_cov = add_sigma_NL(cov, half_sig_NL)
        half_chi2 = get_chi2(c, half_cov)
        half_cdf = sc_chi2.cdf(half_chi2, df)
        if verbose:
            if MPI is None:
                print(prefix+"Optimising")
            else:
                MPI.mpi_print_zero(prefix+"Optimising")
        while abs(half_cdf - 0.5) > etol:
            if half_cdf < 0.5:
                max_sig_NL = half_sig_NL
            else:
                min_sig_NL = half_sig_NL
            half_sig_NL = (max_sig_NL + min_sig_NL)/2.
            half_cov = add_sigma_NL(cov, half_sig_NL)
            half_chi2 = get_chi2(c, half_cov)
            half_cdf = sc_chi2.cdf(half_chi2, df)
            if verbose:
                if MPI is None:
                    print(prefix+"- sigma_NL = %0.2f, chi2 = %0.2f, CDF = %0.2f" % (half_sig_NL, half_chi2, half_cdf))
                else:
                    MPI.mpi_print_zero(prefix+"- sigma_NL = %0.2f, chi2 = %0.2f, CDF = %0.2f" % (half_sig_NL, half_chi2, half_cdf))
        if verbose:
            if MPI is None:
                print(prefix+"Optimised!")
                print(prefix+"sigma_NL = %0.2f, chi2 = %0.2f, CDF = %0.2f" % (half_sig_NL, half_chi2, half_cdf))
            else:
                MPI.mpi_print_zero(prefix+"Optimised!")
                MPI.mpi_print_zero(prefix+"sigma_NL = %0.2f, chi2 = %0.2f, CDF = %0.2f" % (half_sig_NL, half_chi2, half_cdf))
        success = True
        sigma_NL = half_sig_NL
    return success, sigma_NL
