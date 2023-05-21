import numpy as np


def write_constraints(fname, x, y, z, ex, ey, ez, u, u_err):
    """Writes constraints in the npz format for input into MIMIC.

    Parameters
    ----------
    fname : str
        Filename for constraint file.
    x, y, z : array
        Location of constraints.
    ex, ey, ez : array
        Vector showing the direction of the velocity.
    u, u_err : array
        Velocity and associated error.
    """
    if fname[-4:] != '.npz':
        fname += '.npz'
    np.savez(fname, x=x, y=y, z=z, ex=ex, ey=ey, ez=ez, u=u, u_err=u_err)
