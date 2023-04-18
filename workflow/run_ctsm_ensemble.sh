#!/bin/bash
#PBS -A UCUB0089
#PBS -q economy
#PBS -N run_ctsm_ensemble
#PBS -l walltime=12:00:00
#PBS -o /glade/work/yifanc/NNA/optmz/workflow/log/
#PBS -e /glade/work/yifanc/NNA/optmz/workflow/log/
#PBS -l select=1:ncpus=36:mpiprocs=1

./workflow/runens.sh ${ENV_FILE_REP}
