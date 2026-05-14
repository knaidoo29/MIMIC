from os import environ

N_THREADS = '1'

environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS

import sys
import os.path
import numpy as np

from mimic.main import MIMIC
from mimic.ext import mpiutils

MPI = mpiutils.MPI()

yaml_fname = str(sys.argv[1])

mimic = MIMIC(MPI)

mimic.run(yaml_fname)
