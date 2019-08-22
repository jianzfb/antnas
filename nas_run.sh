#!/usr/bin/env bash
python3 nas_main.py -dset CIFAR10 -oc 40000000 -om max -cuda=0,1,2,3,4