#!/bin/bash
#SBATCH --nodes=1                     #Numero de Nós
#SBATCH --ntasks-per-node=1          #Numero de tarefas por Nó
#SBATCH --ntasks=1                  #Numero total de tarefas MPI
#SBATCH -p cpu_dev                 #Fila (partition) a ser utilizada
#SBATCH -J sdumont-mpi                 #Nome job
#SBATCH --exclusive                    #Utilização exclusiva dos nós durante a execução do job
#SBATCH --time=00:01:00

#############################################
#   Para a submissão na Sdumont:            #
#   sbatch sub.sh EXECUTÁVEL                #
#-------------------------------------------#
#   Para a visualização dos Jobs:           #
#   squeue -u $USER                         #
#   ou                                      #
#   watch squeue -u $USER (modo de visão    #
#   tela cheia, para sair CTRL+C)           #
#-------------------------------------------#
#   Para cancelar o Job:                    #
#   scancel NUMERO_DO_JOB                   #
#############################################

source /scratch/app/modulos/intel-psxe-2017.1.043.sh
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so

export OMP_NUMTHREADS=1
EXEC=${1}

time srun -n $SLURM_NTASKS $EXEC 8 8




