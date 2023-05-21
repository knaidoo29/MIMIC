import sys

import mimic
from mimic.ext import mpiutils

MPI = mpiutils.MPI()

yaml_fname = str(sys.argv[1])

MIMIC = mimic.main.MIMIC(MPI)

MIMIC.run(yaml_fname)
