#!/bin/bash

   source /scratch/app/modulos/intel-psxe-2017.1.043.sh
   mpiicc -std=c++11 -qopenmp main_mpi.cpp -o main_mpi

echo "  **** COMPILACAO COMPLETA ****  " 
