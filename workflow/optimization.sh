#!/bin/bash
#PBS -A UCUB0089
#PBS -q premium 
#PBS -N train_surrogate_model
#PBS -l walltime=12:00:00
#PBS -o /glade/work/yifanc/NNA/optmz/workflow/log/
#PBS -e /glade/work/yifanc/NNA/optmz/workflow/log/
#PBS -l select=1:ncpus=36:mpiprocs=1

source activate py3

python ./workflow/scripts/MOASMO_onestep.pe_basin.py ${ITER_REP} 
