import sys
import yaml

from . import files


def read_paramfile(yaml_fname, MPI):
    """Read yaml parameter file.

    Parameters
    ----------
    yaml_fname : str
        Yaml parameter filename.
    MPI : object
        mpi4py object.

    Returns
    -------
    params : dict
        Parameter file dictionary.
    ERROR : bool
        Error tracker.
    """
    MPI.mpi_print_zero(" Reading parameter file: "+yaml_fname)
    if files.check_exist(yaml_fname) is False:
        MPI.mpi_print_zero(" ERROR: Yaml file does not exist, aborting.")
        ERROR = True
    else:
        ERROR = False
    if ERROR is False:
        if MPI.rank == 0:
            with open(yaml_fname, mode="r") as file:
                params = yaml.safe_load(file)
            MPI.send(params, to_rank=None, tag=11)
        else:
            params = MPI.recv(0, tag=11)
        MPI.wait()
        return params, ERROR
    else:
        return 0, ERROR
