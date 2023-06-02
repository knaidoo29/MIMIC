"""setup.py compiles the fortran files so they can be called from python."""

import setuptools
from numpy.distutils.core import setup, Extension

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

exts = []

exts.append(Extension(name='mimic.ext.fiesta.src.grid',
    sources=['mimic/ext/fiesta/src/grid.f90']))
exts.append(Extension(name='mimic.ext.fiesta.src.part2grid_pix',
    sources=['mimic/ext/fiesta/src/part2grid_pix.f90']))
exts.append(Extension(name='mimic.ext.fiesta.src.trilinear',
    sources=['mimic/ext/fiesta/src/trilinear.f90']))

exts.append(Extension(name='mimic.src.interp',
    sources=['mimic/src/interp.f90']))
exts.append(Extension(name='mimic.src.progress',
    sources=['mimic/src/progress.f90']))
exts.append(Extension(name='mimic.src.coords',
    sources=['mimic/src/coords.f90']))
exts.append(Extension(name='mimic.src.fast_correlate',
     sources=['mimic/src/fast_correlate.f90']))
exts.append(Extension(name='mimic.src.fast_correlate_eta',
    sources=['mimic/src/fast_correlate_eta.f90']))
exts.append(Extension(name='mimic.src.fast_correlate_inv',
   sources=['mimic/src/fast_correlate_inv.f90']))

setup(name = 'mimic',
      version = "1.1.1",
      description       = "Model Independent cosMological constrained Initial Conditions",
      long_description  = long_description,
      long_description_content_type = 'text/markdown',
      url               = 'https://knaidoo29.github.io/mistreedoc/',
      author            = "Krishna Naidoo",
      author_email      = "krishna.naidoo.11@ucl.ac.uk",
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=['numpy', 'scipy'],
      ext_modules = exts,
      python_requires = '>=3.4',
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Fortran',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      )
