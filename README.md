# MIMIC: Model Independent cosMological constrained Initial Conditions

|               |                                       |
|---------------|---------------------------------------|
| Author        | Krishna Naidoo                        |
| Version       | 0.0.0-alpha-0                         |
| Repository    | https://github.com/knaidoo29/mimic    |
| Documentation | https://mimic-doc.readthedocs.io/     |


## Dependencies

* [`numpy`](http://www.numpy.org/)
* [`scipy`](https://scipy.org/)
* [`magpie`](https://magpie-doc.readthedocs.io/)

For multiprocessing functionality:

* [`mpi4py`](https://mpi4py.readthedocs.io/)
* [`mpi4py-fft`](https://mpi4py-fft.readthedocs.io/)
* [`MPIutils`](https://github.com/knaidoo29/MPIutils) -- Included in git repository.


## Installation

Clone the git repository and install dependencies. For `MPIutils` enter the
`MPIutils` folder and run

```
python setup.py build
python setup.py install
```

After this is installed then install `mimic` by running

```
python setup.py build
python setup.py install
```

## Support

If you have any issues with the code or want to suggest ways to improve it please
open a new issue ([here](https://github.com/knaidoo29/mimic/issues)) or (if you
don't have a github account) email _krishna.naidoo.11@ucl.ac.uk_.


## Version History

* **Version 0.0**:
    * Constrained simulation functions.
