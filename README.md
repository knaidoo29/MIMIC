# MIMIC: Multiscale Implementation of Mimetic Initial Conditions

`MIMIC` constructs constrained initial conditions and Wiener filter reconstruction from a given set of peculiar velocity constraints and user defined expansion history, power spectra and growth functions. The code is written in `python`, with `MPI` enabled through `mpi4py`, and written with `numba` JIT compiled code for speed.

## Dependencies

The software is dependent on the following libraries that will be installed automatically when installed with `pip`:

* [`numpy`](http://www.numpy.org/)
* [`scipy`](https://scipy.org/)
* [`PyYAML`](https://pyyaml.org/)
* [`fiesta`](https://fiesta-docs.readthedocs.io/)
* [`shift`](https://shift-doc.readthedocs.io/)

To enable `MPI` you must install, note you can still run `MIMIC` without this:
* [`mpi4py`](https://mpi4py.readthedocs.io/)

## Installation

Clone the git repository and install `mimic` by running inside the cloned repository:

```
pip install .
```

You should now be able to import the module:

```python
import mimic
```

## Pipeline

`MIMIC` will be installed as a module, which can be called from any python script or notebook. The `MIMIC` script file, `mimic-run.py` is located in the `scripts/` folder. This file can (and should) be copied to the directory in which you will be running analyses, which you want to compute `MIMIC` data products.

## Support

If you have any issues with the code or want to suggest ways to improve it please open a new issue ([here](https://github.com/knaidoo29/mimic/issues)) or (if you don't have a github account) email _krishna.naidoo.11@ucl.ac.uk_.
