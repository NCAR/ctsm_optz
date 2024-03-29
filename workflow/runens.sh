#!/bin/bash
# Daniel Kennedy (djk2120@ucar.edu)

if [ $# -eq 0 ]
then
    echo "ERROR: please specify format file"
    echo "   ex: ./runens.sh spinAD.env"
    exit 1
fi

#set up environment variables
source $1
jobdir=$(pwd)"/"
echo "route_job_id_file" > ${route_job_id_file}

#collect restarts if needed
if [ "$finidatFlag" = true ]
then
    if [ ! -d $RESTARTS ]
    then
	mkdir $RESTARTS
    fi
    cd $CASE_DIR$prevCase
    for p in $(ls); do
	cd $p
	keyfile=$p"_key.txt"
	d=$SCRATCH$p"/run/"
	while read -r line; do 
	    tmp=(${line///}) 
	    paramkey=${tmp[1]} 
	    instkey=${tmp[0]}
	    oldfile=$d$p".clm2_"$instkey".r.*"
	    newfile=$RESTARTS$paramkey$finidatSuff
	    echo "cp "$oldfile" "$newfile
	    cp $oldfile $newfile
	done < $keyfile
	cd ..
    done
fi

#count existing cases
# so that we give this case a new name
cd $CASE_DIR
if [ -d $casePrefix ]
then
    j=$(ls $casePrefix | wc -w)
else j=0
fi
cd $jobdir

# custom function to read n lines from a file
read_n() { for i in $(seq $1); do read || return; echo $REPLY; done; }

# count how many parameter sets in this batch
nx=$(wc -l < $paramList)

# outer loop creates a new multi-instance case
i=0
while [ $i -lt $nx ]
do
    nl=$((nx-i)) #how many more lines in the file?
    if [ $nl -lt $ninst ]
    then         #adjust ninst if there's an odd remainder
	ninst=$nl
	exerootFlag=false  #must rebuild
    fi

    i=$((i+ninst))
    j=$((j+1))

    #create the new case 
    repcase=$casePrefix"_"$(seq -f "%03g" $j $j)
    echo "--------------------------"
    echo "   creating "$repcase
    echo "--------------------------"
    cd $CASE_DIR
    ${SCRIPTS_DIR}/create_clone --case $casePrefix"/"$repcase --clone $basecase
    cd $casePrefix"/"$repcase
    ./xmlchange NINST_LND=$ninst
    ./case.setup --reset
    if [ "$exerootFlag" = true ]
    then
	./xmlchange BUILD_COMPLETE=TRUE
	./xmlchange EXEROOT=$exeroot
    fi
    
    # start generating routing control files
    route_input="/glade/scratch/yifanc/archive/${repcase}/lnd/hist/"
    route_output="/glade/scratch/yifanc/archive/${repcase}/rof/hist/"
    route_control_list=${source_dir}"routing/control/control/${repcase}.mizuRoute.control.sum.csv"
    dic_repcase_jobid=${source_dir}"routing/control/control/${repcase}.dic_repcase_jobid.csv"

    if [ -f ${route_control_list} ]; then
        rm ${route_control_list}
    fi
    if [ -f ${dic_repcase_jobid} ]; then
        rm ${dic_repcase_jobid}
    fi

    #   creating a user_nl_clm_00xx for each paramset
    CT=0
    lines="$(read_n $ninst)"
    for p in $lines; do 
	CT=$((CT+1))
	printf -v nlnum "%04d" $CT
        if [ $ninst -eq 1 ]; then
            nlfile="user_nl_clm"
        else
	    nlfile="user_nl_clm_"$nlnum
        fi
	pfile=$PARAMS_DIR$p".nc"
	pfilestr="paramfile = '"$pfile"'"

        # make sure the parameter file is cdf5 file
        nccopy -k 5 ${pfile} ${pfile}.cdf5
        mv ${pfile}.cdf5 ${pfile}

	# copy user_nl_clm and specify paramfile
	cd $CASE_DIR$casePrefix"/"$repcase
	cp user_nl_clm $nlfile
	echo $pfilestr >> $nlfile

	# cat nlmods if needed
	if [ "$nlmodsFlag" = true ]
	then
	    nlmods=$NLMODS_DIR$p".txt"
	    cat $nlmods >> $nlfile
	fi

	# specify finidat if needed
	if [ "$finidatFlag" = true ]
	then
	    rfile=$RESTARTS$p$finidatSuff
	    rfilestr="finidat ='"$rfile"'"
	    echo $rfilestr >> $nlfile
	fi

	# create a key to map each instance number to its paramfile
	printf $nlnum"\t"$p"\n" >> $repcase"_key.txt"

        # generate control file for routing
        route_control_file="${repcase}.clm2_${nlnum}.mizuRoute.control"
        route_input_file="${repcase}.clm2_${nlnum}.${CTSM_RUNOFF_ID}.${SY}-${SM}-${SD}-00000.nc"
        route_output_case="${repcase}.clm2_${nlnum}"
        
        ${ROUTE_CONTROL_SCRIPT} ${route_control_file} ${route_input} ${route_output} ${route_input_file} ${route_output_case} ${basin}
        echo ${route_control_file} >> ${route_control_list}
        echo ${repcase} ${nlnum} ${p}_${basin} >> ${dic_repcase_jobid}

    done

    if [ "$exerootFlag" = false ]; then
	echo "--------------------------"
	echo "   building "$repcase
	echo "--------------------------"
	./case.build
	#only need to compile the source code once
	exeroot=$SCRATCH$repcase"/bld"
	exerootFlag=true
    fi

    echo "--------------------------"
    echo "   submitting "$repcase
    echo "--------------------------"
    ctsm_job_id_file=${source_dir}"routing/qsub/log/ctsm_job_id.${repcase}"
    ./case.submit > ${ctsm_job_id_file}
    
    # echo getting the CTSM job ID
    ctsm_job_id=$(grep -oP '(?<=Submitted job case.st_archive with id )[0-9]+' ${ctsm_job_id_file})    

    # echo submitting routing jobs
    route_job_id=$(qsub -v CONTROL_FILE_SUM=${route_control_list},JOB_FILE=${dic_repcase_jobid},METRIC_OUTPUT_DIR=${OBJ_OUT_DIR},FLOW_DIR=${route_output},SWE_DIR=${route_input},HRU_SHP=${HRU_SHP},SIM_DOMAIN_FILE=${SIM_DOMAIN},MAPPING_FILE=${FLOW_MAPPING_FILE} -W depend=afterok:${ctsm_job_id} ${ROUTE_QSUB_SCRIPT})
    echo ${route_job_id} >> ${route_job_id_file} 
#    qsub -v CONTROL_FILE_SUM=${route_control_list},JOB_FILE=${dic_repcase_jobid},METRIC_OUTPUT_DIR=${OBJ_OUT_DIR},FLOW_DIR=${route_output},SWE_DIR=${route_input},HRU_SHP=${HRU_SHP},SIM_DOMAIN_FILE=${SIM_DOMAIN},MAPPING_FILE=${FLOW_MAPPING_FILE}  ${ROUTE_QSUB_SCRIPT}

done < $paramList


