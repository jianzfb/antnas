#!/usr/bin/env bash
architecture_folder=""
architecture_name_list=(1 2 3)
architecture_cuda_list=(0 0 0)
index=1
for item in ${architecture_name_list[@]}
do
    python $0 -cuda=${#architecture_cuda_list[index]} -draw_env=PK_${index} -architecture=${architecture_folder}${item}
    index=`expr $index + 1`
done
