import yaml


from . import check, error


def _nompi_read_paramfile(yaml_fname):
    """Read yaml parameter file with no mpi.

    Parameters
    ----------
    yaml_fname : str
        Yaml parameter filename.

    Returns
    -------
    params : dict
        Parameter file dictionary.
    ERROR : bool
        Error tracker.
    """
    print(" Reading parameter file: "+yaml_fname)
    if check.isfile(yaml_fname) is False:
        error._error_message(ERROR, "Yaml file does not exist.")
        ERROR = True
    else:
        ERROR = False
    if ERROR is False:
        params = yaml.safe_load(file)
        return params, ERROR
    else:
        return 0, ERROR


def _mpi_read_paramfile(yaml_fname, MPI):
    """Read yaml parameter file.

    Parameters
    ----------
    yaml_fname : str
        Yaml parameter filename.
    MPI : object
        MPIutils mpi4py object.

    Returns
    -------
    params : dict
        Parameter file dictionary.
    ERROR : bool
        Error tracker.
    """
    MPI.mpi_print_zero(" Reading parameter file: "+yaml_fname)
    if check.isfile(yaml_fname) is False:
        error._error_message(ERROR, "Yaml file does not exist.", MPI=MPI)
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


def read_paramfile(yaml_fname, MPI=None):
    """Read yaml parameter file.

    Parameters
    ----------
    yaml_fname : str
        Yaml parameter filename.
    MPI : object, optional
        MPIutils mpi4py object.

    Returns
    -------
    params : dict
        Parameter file dictionary.
    ERROR : bool
        Error tracker.
    """
    if MPI is None:
        return _nompi_read_paramfile(yaml_fname)
    else:
        return _mpi_read_paramfile(yaml_fname, MPI)
