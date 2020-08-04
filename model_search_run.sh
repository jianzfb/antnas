#!/usr/bin/env bash
python portrait_train_and_search.py -exp-project nas2 -exp-name v2 -arch PKBiSegSN2 -bs 128 -epochs 1200 -evo_epochs 80 -dset SEG -path ./portrait_dataset/refine -population_size 50 -cuda 0,1 -lr_decay cos -optim SGD