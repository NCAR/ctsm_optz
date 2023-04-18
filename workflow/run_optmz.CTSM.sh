#!/bin/bash
#PBS -A UCUB0089
#PBS -q economy
#PBS -N ASMO
#PBS -l walltime=12:00:00
#PBS -o /glade/work/yifanc/NNA/optmz/workflow/log/
#PBS -e /glade/work/yifanc/NNA/optmz/workflow/log/
#PBS -l select=1:ncpus=36:mpiprocs=1

source activate py3

optimization_qsub_file="./workflow/optimization.sh"
env_temp_file="./workflow/env/pe_basin.OM.temp.env"
run_ctsm_ensemble_file="./workflow/run_ctsm_ensemble.sh"
num_resample=20
ninst=5
num_lines_route=$(echo $((num_resample / ninst + 1)))
iter_start=1
iter_end=5

for iter in $(seq $iter_start $iter_end); do # {2..5}; do
    for basin in talkeetna salcha beaver; do #tanana kuparuk 
        route_job_id_file="./workflow/temp_file/route_job_id.iter_${iter}.${basin}.csv"
        if [ -f ${route_job_id_file} ]; then
            rm ${route_job_id_file}
        fi
        env_file="./workflow/env/pe_basin_${iter}.${basin}.env"
        sed "s#BASIN_REP#${basin}#g" ${env_temp_file} > ${env_file}
        sed -i "s#ITER_REP#${iter}#g" ${env_file}
        if [ "$iter" = $iter_start ]; then
            job1=$(qsub -v ENV_FILE_REP=${env_file} ${run_ctsm_ensemble_file})
        else
            job1=$(qsub -v ENV_FILE_REP=${env_file} -W depend=afterok:${job2} ${run_ctsm_ensemble_file})
        fi
    done
    until [ $(wc -l < ${route_job_id_file}) == ${num_lines_route} ]
    do
        sleep 5
    done
    route_job_id_list=""
    for basin in talkeetna salcha beaver; do
        route_job_id_file="./workflow/temp_file/route_job_id.iter_${iter}.${basin}.csv"
        i=1 n=0
        while read -r line; do
            ((n >= i )) && route_job_id_list=${route_job_id_list}:$(echo ${line} | awk -F. '{print $1}')
            ((n++))
        done < ${route_job_id_file}
        echo ${route_job_id_list}
    done
    job2=$(qsub -v ITER_REP=${iter} -W depend=afterok${route_job_id_list} ${optimization_qsub_file})
#    qsub -v ITER_REP=${iter} ${optimization_qsub_file}
done
#/glade/u/home/yifanc/code/CLM5PPE/jobscripts/runens.sh Tanana_0.env 
#/glade/u/home/yifanc/code/CLM5PPE/jobscripts/runens.sh pe_basin_0.kuparuk.env
#/glade/u/home/yifanc/code/CLM5PPE/jobscripts/runens.routing_only.sh pe_basin_0.tanana.env
#    last_route_job_id=$(echo $(tail -1 ${route_job_id_file}) | awk -F. '{print $1}')
#    echo ${last_route_job_id}
