#!/usr/bin/env bash
architecture_folder=""
architecture_name_list=(1 2 3)
index=1
for item in ${architecture_name_list[@]}
do
    python train.py -cuda=0 -draw_env=PK_${index} -architecture=${architecture_folder}${item}
    index=`expr $index + 1`
done
