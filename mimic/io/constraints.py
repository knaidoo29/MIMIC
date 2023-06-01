import numpy as np

from . import check, error


def _load_constraints_ice(fname):
    """Loads an ICeCoRe constraint file.

    Parameters
    ----------
    fname : str
        Filename of the constraint file.

    Returns
    -------
    x, y, z : array
        Location of constraints.
    ex, ey, ez : array
        Vector showing the direction of the velocity.
    c, c_err : array
        Velocity and associated error.
    c_type : array
        Constraint type: 0 - delta, 1 - phi, 2 - vel.
    """
    # Sanity checks
    ERROR = error._error_if_false(check.isfile(fname))
    error._error_message(ERROR, "File %s does not exist"%fname)
    error._break4error(ERROR)
    # Load data file
    data = np.loadtxt(fname, unpack=True)
    c_type = data[0]-1
    x, y, z = data[1], data[2], data[3]
    c, c_err = data[4], data[5]
    ex, ey, ez = data[6], data[7], data[8],
    RG = data[9]
    return x, y, z, ex, ey, ez, c, c_err, c_type


def _save_constraints_ice(fname, x, y, z, ex, ey, ez, c, c_err, c_type):
    """Saves an ICeCoRe constraint file.

    Parameters
    ----------
    fname : str
        Filename for constraint file.
    x, y, z : array
        Location of constraints.
    ex, ey, ez : array
        Vector showing the direction of the velocity.
    c, c_err : array
        Velocity and associated error.
    c_type : array
        Constraint type: 0 - delta, 1 - phi, 2 - vel.
    """
    cons_type = np.copy(c_type) + 1
    cons_type = cons_type.astype('int')
    RG = np.zeros(len(x))
    data = np.column_stack([cons_type, x, y, z, u, u_err, ex, ey, ez, RG])
    np.savetxt(fname, data, fmt="%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f")


def _load_constraints_npz(fname):
    """Writes constraints in the npz format for input into MIMIC.

    Parameters
    ----------
    fname : str
        Filename for constraint file.

    Returns
    -------
    x, y, z : array
        Location of constraints.
    ex, ey, ez : array
        Vector showing the direction of the velocity.
    c, c_err : array
        Velocity and associated error.
    c_type : array
        Constraint type: 0 - delta, 1 - phi, 2 - vel.
    """
    if fname[-4:] != '.npz':
        fname += '.npz'
    # Sanity checks
    ERROR = error._error_if_false(check.isfile(fname))
    error._error_message(ERROR, "File %s does not exist"%fname)
    error._break4error(ERROR)
    data = np.load(fname)
    x, y, z = data['x'], data['y'], data['z']
    ex, ey, ez = data['ex'], data['ey'], data['ez']
    c, c_err, c_type = data['c'], data['c_err'], data['c_type'].astype('int')
    return x, y, z, ex, ey, ez, c, c_err, c_type


def _save_constraints_npz(fname, x, y, z, ex, ey, ez, c, c_err, c_type):
    """Writes constraints in the npz format for input into MIMIC.

    Parameters
    ----------
    fname : str
        Filename for constraint file.
    x, y, z : array
        Location of constraints.
    ex, ey, ez : array
        Vector showing the direction of the velocity.
    c, c_err : array
        Constraints and associated error.
    c_type : array
        Constraint type: 0 - delta, 1 - phi, 2 - vel.
    """
    if fname[-4:] != '.npz':
        fname += '.npz'
    np.savez(fname, x=x, y=y, z=z, ex=ex, ey=ey, ez=ez, c=c, c_err=c_err, c_type=c_type)


def load_constraints(fname, filetype='npz'):
    """Writes constraints file in either npz or ICeCoRe format.

    Parameters
    ----------
    fname : str
        Filename for constraint file.
    x, y, z : array
        Location of constraints.
    ex, ey, ez : array
        Vector showing the direction of the velocity.
    c, c_err : array
        Velocity and associated error.
    c_type : array
        Constraint type: 0 - delta, 1 - phi, 2 - vel.
    filetype : str, optional
        - 'npz' for numpy binary format.
        - 'ice' for ICeCoRe format.
    """
    # Sanity checks
    ERROR = error._error_if_false(check.isscalar(filetype))
    error._error_message(ERROR, 'filetype cannot be an array')
    error._break4error(ERROR)
    # Load file
    if filetype == 'npz':
        return _load_constraints_npz(fname)
    elif filetype == 'ice':
        return _load_constraints_ice(fname)


def save_constraints(fname, x, y, z, ex, ey, ez, c, c_err, c_type, filetype='npz'):
    """Writes constraints file in either npz or ICeCoRe format.

    Parameters
    ----------
    fname : str
        Filename for constraint file.
    x, y, z : array
        Location of constraints.
    ex, ey, ez : array
        Vector showing the direction of the velocity.
    c, c_err : array
        Velocity and associated error.
    c_type : array
        Constraint type: 0 - delta, 1 - phi, 2 - vel.
    filetype : str, optional
        - 'npz' for numpy binary format.
        - 'ice' for ICeCoRe format.
    """
    # Sanity checks
    ERROR = error._error_if_false(check.isscalar(filetype))
    error._error_message(ERROR, 'filetype cannot be an array')
    error._break4error(ERROR)
    # Load file
    if filetype == 'npz':
        _save_constraints_npz(fname, x, y, z, ex, ey, ez, c, c_err, c_type)
    elif filetype == 'ice':
        _save_constraints_ice(fname, x, y, z, ex, ey, ez, c, c_err, c_type)
