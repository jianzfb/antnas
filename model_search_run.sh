#!/usr/bin/env bash
python -u imagenet_train_and_search.py -exp-project dual -exp-name v2 -arch PKAsynImageNetSN -bs 384 -epochs 150 -evo_epochs 0 -dset ImageNetV2 -path ./portrait_dataset/ -population_size 50 -cuda 0,1,2,3,4,5,6,7 -lr_decay cos -optim SGD
