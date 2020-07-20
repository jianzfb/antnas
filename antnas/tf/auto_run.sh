#!/bin/bash
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#nohup python -u cifar10_train.py -arch epoch_360_accuray_0.0609_latency_21.99_params_908576.architecture -model_dir with_rate > log_train 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u train.py -arch epoch_525_accuray_0.0640_latency_24.23_params_699456.architecture -model_dir checkpoint0
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u train.py -arch epoch_525_accuray_0.0641_latency_20.01_params_699456.architecture -model_dir checkpoint1
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u train.py -arch epoch_525_accuray_0.0645_latency_21.38_params_699456.architecture -model_dir checkpoint2
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u train.py -arch epoch_525_accuray_0.0646_latency_22.89_params_699456.architecture -model_dir checkpoint3
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u train.py -arch epoch_525_accuray_0.0648_latency_22.77_params_699456.architecture -model_dir checkpoint4
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u train.py -arch epoch_525_accuray_0.0651_latency_29.17_params_699456.architecture -model_dir checkpoint5
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u train.py -arch epoch_525_accuray_0.0657_latency_34.61_params_699456.architecture -model_dir checkpoint6
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u train.py -arch epoch_525_accuray_0.0659_latency_43.78_params_699456.architecture -model_dir checkpoint7
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u train.py -arch epoch_525_accuray_0.0670_latency_24.31_params_699456.architecture -model_dir checkpoint8
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u train.py -arch epoch_525_accuray_0.0674_latency_14.90_params_699456.architecture -model_dir checkpoint9

#tail -f log_train
