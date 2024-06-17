#!/bin/bash

# This script will download external module files from fiesta, magpie, mpiutils and shift
# to be used internally in cactus and avoid unnecessary dependencies.

echo " "
echo " Synchronising external modules"
echo " "

echo " Downloading FIESTA files"
echo " "

cd mimic/ext/fiesta

fiestalist="mpi_periodic periodic"

cd boundary

for fiestafile in $fiestalist ;
  do
    echo " "
    echo " Downloading ${fiestafile}.py from https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/boundary/${fiestafile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/boundary/${fiestafile}.py
    if test -f ${fiestafile}.py.1; then
      echo " "
      echo " ext/fiesta/boundary/${fiestafile}.py was downloaded"
      echo " "
      rm ${fiestafile}.py
      mv ${fiestafile}.py.1 ${fiestafile}.py
    else
      echo " "
      echo " ERROR: ext/fiesta/boundary/${fiestafile}.py was not downloaded!"
      echo " "
    fi
  done

cd ..

fiestalist="mpi_points points"

cd coords

for fiestafile in $fiestalist ;
  do
    echo " "
    echo " Downloading ${fiestafile}.py from https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/coords/${fiestafile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/coords/${fiestafile}.py
    if test -f ${fiestafile}.py.1; then
      echo " "
      echo " ext/fiesta/coords/${fiestafile}.py was downloaded"
      echo " "
      rm ${fiestafile}.py
      mv ${fiestafile}.py.1 ${fiestafile}.py
    else
      echo " "
      echo " ERROR: ext/fiesta/coords/${fiestafile}.py was not downloaded!"
      echo " "
    fi
  done

cd ..

fiestalist="trilinear"

cd dtfe

for fiestafile in $fiestalist ;
  do
    echo " "
    echo " Downloading ${fiestafile}.py from https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/interp/${fiestafile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/dtfe/${fiestafile}.py
    if test -f ${fiestafile}.py.1; then
      echo " "
      echo " ext/fiesta/interp/${fiestafile}.py was downloaded"
      echo " "
      rm ${fiestafile}.py
      mv ${fiestafile}.py.1 ${fiestafile}.py
    else
      echo " "
      echo " ERROR: ext/fiesta/interp/${fiestafile}.py was not downloaded!"
      echo " "
    fi
  done

cd ..

fiestalist="grid part2grid_pix trilinear"

cd src

for fiestafile in $fiestalist ;
  do
    echo " "
    echo " Downloading ${fiestafile}.py from https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/src/${fiestafile}.f90"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/src/${fiestafile}.f90
    if test -f ${fiestafile}.f90.1; then
      echo " "
      echo " ext/fiesta/src/${fiestafile}.f90 was downloaded"
      echo " "
      rm ${fiestafile}.f90
      mv ${fiestafile}.f90.1 ${fiestafile}.f90
    else
      echo " "
      echo " ERROR: ext/fiesta/src/${fiestafile}.f90 was not downloaded!"
      echo " "
    fi
  done

cd ../../../..


echo " Downloading MPIutils files"
echo " "

cd mimic/ext/mpiutils

mpiutilslist="loops mpiclass"

for mpiutilsfile in $mpiutilslist ;
  do
    echo " "
    echo " Downloading ${mpiutilsfile}.py from https://raw.githubusercontent.com/knaidoo29/MPIutils/master/mpiutils/${mpiutilsfile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/MPIutils/master/mpiutils/${mpiutilsfile}.py
    if test -f ${mpiutilsfile}.py.1; then
      echo " "
      echo " ext/mpiutils/${mpiutilsfile}.py was downloaded"
      echo " "
      rm ${mpiutilsfile}.py
      mv ${mpiutilsfile}.py.1 ${mpiutilsfile}.py
    else
      echo " "
      echo " ERROR: ext/mpiutils/${mpiutilsfile}.py was not downloaded!"
      echo " "
    fi
  done

cd ../../..


echo " Downloading SHIFT files"
echo " "

cd mimic/ext/shift/cart

shiftlist="grid kgrid mpi_fft mpi_grid mpi_kgrid utils"

for shiftfile in $shiftlist ;
  do
    echo " "
    echo " Downloading ${shiftfile}.py from https://raw.githubusercontent.com/knaidoo29/SHIFT/master/shift/cart/${shiftfile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/SHIFT/master/shift/cart/${shiftfile}.py
    if test -f ${shiftfile}.py.1; then
      echo " "
      echo " ext/shift/cart/${shiftfile}.py was downloaded"
      echo " "
      rm ${shiftfile}.py
      mv ${shiftfile}.py.1 ${shiftfile}.py
    else
      echo " "
      echo " ERROR: ext/shift/cart/${shiftfile}.py was not downloaded!"
      echo " "
    fi
  done

cd ../../../..

editfiles="fiesta/coords/mpi_points"

echo " "
echo " Editing files with 'import shift'"
echo " "

for editfile in $editfiles ;
  do
    echo " Editing 'import shift' -> 'from ... import shift' in file cactus/ext/${editfile}.py"
    sed -i '' 's/import shift/from ... import shift/' cactus/ext/${editfile}.py
  done

echo " "
