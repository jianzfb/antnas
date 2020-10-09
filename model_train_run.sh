#!/usr/bin/env bash
exp_project="baiduportrait"
architecture_folder="./suggestion-1"
cuda=''
optim='RMS'
index=1
lr=0.0001
dset='ImageNetV2'
bs=256
epochs=150
for file in `ls ${architecture_folder}`
do
    if [[ "${file}" =~ .*architecture$ ]]
    then
      echo ${file}
      python $1 -cuda=${cuda} -exp-project=${exp_project} -exp-name=PK_${index} -architecture=${architecture_folder}/${file} -optim=${optim} -lr=${lr} -dset=${dset} -bs=${bs} -epochs=${epochs}
      index=`expr $index + 1`
    fi
done