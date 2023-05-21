import numpy as np
from .. import src


def trilinear(fgrid, boxsize, x, y, z, origin=0., fill_value=np.nan, periodic=True):
    """ Trilinear interpolation from a 3D grid defined in box of [0., boxsize].

    Parameter
    ---------
    fgrid : array
        Field values on a 3D grid.
    boxsize : float/array
        Box size in one or all axes.
    x : array
        x coordinate values.
    y : array
        y coordinate values.
    z : array
        z coordinate values.
    origin : float/array
        Origin for the axes.
    fill_value : float, optional
        Fill outside boundary values.
    periodic : bool, optional
        Determines whether to interpolate on a periodic grid.

    Returns
    -------
    f : array
        Field interpolation values.
    """
    # determine ngrid from fgrid
    ngrids = np.shape(fgrid)
    if np.isscalar(boxsize):
        xbox = boxsize
        ybox = boxsize
        zbox = boxsize
    else:
        xbox, ybox, zbox = boxsize[0], boxsize[1], boxsize[2]
    if np.isscalar(origin):
        _x = np.copy(x) - origin
        _y = np.copy(y) - origin
        _z = np.copy(z) - origin
    else:
        _x = np.copy(x) - origin[0]
        _y = np.copy(y) - origin[1]
        _z = np.copy(z) - origin[2]
    # check if particles are inside the box
    cond = np.where((_x >= 0.) & (_x < xbox) & (_y >= 0.) & (_y < ybox)
                    & (_z >= 0.) & (_z < zbox))[0]
    if len(cond) == len(_x):
        # All particles are within the boundaries so no boundary management is necessary.
        npart = len(_x)
        if np.isscalar(periodic):
            if periodic == True:
                f = src.trilinear_periodic(fgrid=fgrid.flatten(), x=_x, y=_y, z=_z,
                                           xbox=xbox, ybox=ybox, zbox=zbox,
                                           ngridx=ngrids[0], ngridy=ngrids[1],
                                           ngridz=ngrids[2], npart=npart)
            else:
                f = src.trilinear_nonperiodic(fgrid=fgrid.flatten(), x=_x, y=_y, z=_z,
                                              xbox=xbox, ybox=ybox, zbox=zbox,
                                              ngridx=ngrids[0], ngridy=ngrids[1],
                                              ngridz=ngrids[2], npart=npart)
        else:
            if periodic[0] is True:
                perix = 1
            else:
                perix = 0
            if periodic[1] is True:
                periy = 1
            else:
                periy = 0
            if periodic[2] is True:
                periz = 1
            else:
                periz = 0
            f = src.trilinear_axisperiodic(fgrid=fgrid.flatten(), x=_x, y=_y, z=_z,
                                           xbox=xbox, ybox=ybox, zbox=zbox,
                                           perix=perix, periy=periy, periz=periz,
                                           ngridx=ngrids[0], ngridy=ngrids[1],
                                           ngridz=ngrids[2], npart=npart)
    else:
        # Some particles are outside the boundary.
        # create a mask for in and outside the box
        mask = np.zeros(len(_x))
        # assign particles in the boundary a binary mask of 1.
        mask[cond] = 1.
        # find trilinear interpolation for points inside the boundary.
        npart = len(x[cond])
        f = np.zeros(len(_x))
        if np.isscalar(periodic):
            if periodic == True:
                f[cond] = src.trilinear_periodic(fgrid=fgrid.flatten(), x=_x[cond],
                                                 y=_y[cond], z=_z[cond], xbox=xbox,
                                                 ybox=ybox, zbox=zbox,
                                                 ngridx=ngrids[0], ngridy=ngrids[1],
                                                 ngridz=ngrids[2], npart=npart)
            else:
                f[cond] = src.trilinear_nonperiodic(fgrid=fgrid.flatten(),
                                                    x=_x[cond], y=_y[cond],
                                                    z=_z[cond], xbox=xbox,
                                                    ybox=ybox, zbox=zbox,
                                                    ngridx=ngrids[0], ngridy=ngrids[1],
                                                    ngridz=ngrids[2], npart=npart)
        else:
            if periodic[0] is True:
                perix = 1
            else:
                perix = 0
            if periodic[1] is True:
                periy = 1
            else:
                periy = 0
            if periodic[2] is True:
                periz = 1
            else:
                periz = 0
            f[cond] = src.trilinear_axisperiodic(fgrid=fgrid.flatten(), x=_x[cond],
                                                 y=_y[cond], z=_z[cond], xbox=xbox,
                                                 ybox=ybox, zbox=zbox, perix=perix,
                                                 periy=periy, periz=periz, ngridx=ngrids[0],
                                                 ngridy=ngrids[1], ngridz=ngrids[2], npart=npart)
        # fill outside boundary with fill values.
        cond = np.where(mask == 0.)[0]
        f[cond] = fill_value
    return f
