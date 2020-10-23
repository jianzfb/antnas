#!/usr/bin/env bash
python -u $1 -exp-project dual -exp-name v1 -arch PKAsynImageNetSN -bs 384 -epochs 0 -evo_epochs 50 -dset ImageNetV2 -path ./portrait_dataset/ -model_path=./supernetwork/supernetwork_state_0.supernet.model -population_size 50 -cuda 0 -lr_decay cos -optim SGD
