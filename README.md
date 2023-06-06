# MIMIC: Model Independent cosMological constrained Initial Conditions

|               |                                       |
|---------------|---------------------------------------|
| Author        | Krishna Naidoo                        |
| Version       | 1.1.1                                 |
| Repository    | https://github.com/knaidoo29/mimic    |
| Documentation | https://mimic-doc.readthedocs.io/     |

## Dependencies

* [`mpi4py`](https://mpi4py.readthedocs.io/)
* [`mpi4py-fft`](https://mpi4py-fft.readthedocs.io/)
* [`numpy`](http://www.numpy.org/)
* [`scipy`](https://scipy.org/)
* PyYAML -- link actual module page.
* [`MPIutils`](https://github.com/knaidoo29/MPIutils) -- Not to be confused with the pip installable mpiutils, but this package specifically. -- Need to sort this out so there won't be any confusion.

Note, some functions have been copied from following libraries and are stored in the mimic/ext/ subdirectory.
* [`fiesta`](https://fiesta-docs.readthedocs.io/)
* [`shift`](https://shift-doc.readthedocs.io/)

## Installation

Clone the git repository and install `mimic` by running

```
python setup.py build
python setup.py install
```
## Pipeline

`MIMIC` will be installed as a module, which can be called from any python script, notebook, etc. The actualy `MIMIC` script file, `mimic-run.py` is located in the scripts/ folder. This file can (and should) be copied to the directory which you want to compute `MIMIC` data products.



## Support

If you have any issues with the code or want to suggest ways to improve it please open a new issue ([here](https://github.com/knaidoo29/mimic/issues)) or (if you don't have a github account) email _krishna.naidoo.11@ucl.ac.uk_.

## Version History

* **Version 1.0**:
  * Model independent cosmological constrained initial conditions and Wiener filter reconstruction from a given set of peculiar velocity constraints and user defined expansion history, power spectra and growth functions.
