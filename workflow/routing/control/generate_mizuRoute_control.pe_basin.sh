#!/bin/bash

declare -A outlet_id
outlet_id=( ["tanana"]="81015538" \
            ["whitehorse"]="81030831" \
            ["kuparuk"]="81000403" \
            ["colville"]="81001684" \
            ["kenai"]="81030914" \
            ["talkeetna"]="81023793" \
            ['salcha']="81015621" \
            ['beaver']="81016865" \
            ['all']="99999999")


mr_control_temp_file="/glade/work/yifanc/NNA/optmz/routing/control/asmo_pe_basin.run.control.temp"
mr_control_out_dir="/glade/work/yifanc/NNA/optmz/routing/control/control/"
mr_control_file=${mr_control_out_dir}${1}

sed "s#MIZU_INPUT_DIRR#${2}#g" ${mr_control_temp_file} > ${mr_control_file}
sed -i "s#MIZU_OUTPUT_DIRR#${3}#g" ${mr_control_file}
sed -i "s#MIZU_INPUT_FILE_NAME#${4}#g" ${mr_control_file} 
sed -i "s#MIZU_OUTPUT_CASE_NAME#${5}#g" ${mr_control_file}
sed -i "s#MIZU_OUTLET_ID#${outlet_id[${6}]}#g" ${mr_control_file}
sed -i "s#MIZU_BASIN#${6}#g"  ${mr_control_file}
