#!/usr/bin/env bash
python nas_main.py -dset CIFAR10 -arch ConvolutionalNeuralFabric -bs 2 -oc 40000000 -om max -cuda=0,1,2,3,4