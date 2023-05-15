import os.path


def check_exist(fname):
    """Checks arameter file exist.

    Parameters
    ----------
    fname : str
        Yaml parameter filename.
    """
    return os.path.exists(fname)
