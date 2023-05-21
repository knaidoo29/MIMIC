# MIMIC: Model Independent cosMological constrained Initial Conditions

|               |                                       |
|---------------|---------------------------------------|
| Author        | Krishna Naidoo                        |
| Version       | 1.0.0                                 |
| Repository    | https://github.com/knaidoo29/mimic    |
| Documentation | https://mimic-doc.readthedocs.io/     |

## Dependencies

* [`numpy`](http://www.numpy.org/)
* [`scipy`](https://scipy.org/)
* [`mpi4py`](https://mpi4py.readthedocs.io/)
* [`mpi4py-fft`](https://mpi4py-fft.readthedocs.io/)

Note, some functions have been copied from following libraries and are stored in the mimic/ext/ subdirectory.
* [`MPIutils`](https://github.com/knaidoo29/MPIutils)
* [`fiesta`](https://fiesta-docs.readthedocs.io/)
* [`shift`](https://shift-doc.readthedocs.io/)

## Installation

Clone the git repository and install `mimic` by running

```
python setup.py build
python setup.py install
```

## Support

If you have any issues with the code or want to suggest ways to improve it please open a new issue ([here](https://github.com/knaidoo29/mimic/issues)) or (if you don't have a github account) email _krishna.naidoo.11@ucl.ac.uk_.


## Version History

* **Version 1.0**:
  * Model independent cosmological constrained initial conditions and Wiener filter reconstruction from a given set of peculiar velocity constraints and user defined expansion history, power spectra and growth functions.
