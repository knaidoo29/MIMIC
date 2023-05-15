import sys

import mimic
import mpiutils

MPI = mpiutils.MPI()

yaml_fname = str(sys.argv[1])

MIMIC = mimic.main.MIMIC(MPI)

MIMIC.run(yaml_fname)
