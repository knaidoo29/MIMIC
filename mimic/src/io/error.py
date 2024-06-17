

def _error_if_true(value):
    """Return error is True if value is True."""
    if value is True:
        return True
    else:
        return False


def _error_if_false(value):
    """Return error is True if value is False."""
    if value is False:
        return True
    else:
        return False


def _error_message(ERROR, message, MPI=None):
    """Prints error message, if ERROR=True.

    Parameters
    ----------
    ERROR : bool
        If true an error has occured.
    message : str
        Error message.
    MPI : object, optional
        MPIutils mpi4py object.
    """
    if ERROR is True:
        if MPI is None:
            print(" ERROR:", message)
        else:
            MPI.mpi_print_zero(" ERROR:", message)


def _break4error(ERROR):
    """Breaks if ERROR is True."""
    if ERROR is True:
        exit()
