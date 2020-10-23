#!/usr/bin/env bash
exp_project="imagenet"
architecture_folder="./suggestion-1"
cuda=''
optim='RMS'
index=1
lr=0.05
lr_decay='cos'
dset='ImageNetV2'
bs=256
epochs=150
weight_decay=0.00004
for file in `ls ${architecture_folder}`
do
    if [[ "${file}" =~ .*architecture$ ]]
    then
      echo ${file}
      python -u $1 -cuda=${cuda} -exp-project=${exp_project} -exp-name=PK_${index} -architecture=${architecture_folder}/${file} -optim=${optim} -lr=${lr} -lr_decay=${lr_decay} -weight_decay=${weight_decay} -dset=${dset} -bs=${bs} -epochs=${epochs}
      index=`expr $index + 1`
    fi
done