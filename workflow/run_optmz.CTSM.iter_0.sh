#!/bin/bash
#PBS -A UCUB0089
#PBS -q economy
#PBS -N ASMO
#PBS -l walltime=12:00:00
#PBS -o /glade/work/yifanc/NNA/optmz/workflow/log/
#PBS -e /glade/work/yifanc/NNA/optmz/workflow/log/
#PBS -l select=1:ncpus=36:mpiprocs=1

source activate py3

whether_test=False #False # True
run_ctsm_ensemble_file="workflow/run_ctsm_ensemble.sh"
env_dir="workflow/env"

for iter in 0; do
    for basin in talkeetna salcha beaver kuparuk; do #tanana kuparuk 

        if [ ${whether_test} = True ]; then
            echo running test
            env_temp_file=${env_dir}/"pe_basin.OM.temp.test.env"
            env_file=${env_dir}/"pe_basin_${iter}.${basin}.test.env"
        else
            echo running real case
            env_temp_file=${env_dir}/"pe_basin.OM.temp.env"
            env_file=${env_dir}/"pe_basin_${iter}.${basin}.env"
        fi
        sed "s#BASIN_REP#${basin}#g" ${env_temp_file} > ${env_file}
        sed -i "s#ITER_REP#${iter}#g" ${env_file}
        qsub -v ENV_FILE_REP=${env_file} ${run_ctsm_ensemble_file}
    done
done
