import os.path
import numpy as np


def isfile(fname):
    """Checks whether a file exist.

    Parameters
    ----------
    fname : str
        Filename.
    """
    return os.path.exists(fname)


def isfolder(folder):
    """Check folder exists.

    Parameters
    ----------
    folder : str
        Folder string.
    """
    return os.path.isdir(folder)



def isscalar(x):
    """More general isscalar function to prevent 0 dimensional numpy arrays
    from being misidentified as arrays even though they are actually scalar
    variables.
    """
    if type(x).__module__ == np.__name__:
        if len(x.shape) == 0:
            return True
        else:
            return False
    else:
        return np.isscalar(x)


def inlist(x, xlist):
    """Checks whether element x is in list.
    """
    if x in xlist:
        return True
    else:
        return False


def bool2yesno(x):
    """Converts boolean to Yes/No string."""
    if x is True:
        return "Yes"
    else:
        return "No"
