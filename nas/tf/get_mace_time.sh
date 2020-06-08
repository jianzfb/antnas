#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 0 -model_dir hsse65
python -u transform_ckpt_to_pb.py -arch 0 -model_dir hsse65
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 1 -model_dir hsse63
python -u transform_ckpt_to_pb.py -arch 1 -model_dir hsse63
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 2 -model_dir hsse35
python -u transform_ckpt_to_pb.py -arch 2 -model_dir hsse35
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 3 -model_dir hsse33
python -u transform_ckpt_to_pb.py -arch 3 -model_dir hsse33
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 4 -model_dir hs65
python -u transform_ckpt_to_pb.py -arch 4 -model_dir hs65
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 5 -model_dir hs63
python -u transform_ckpt_to_pb.py -arch 5 -model_dir hs63
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 6 -model_dir hs35
python -u transform_ckpt_to_pb.py -arch 6 -model_dir hs35
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 7 -model_dir hs33
python -u transform_ckpt_to_pb.py -arch 7 -model_dir hs33
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 8 -model_dir se65
python -u transform_ckpt_to_pb.py -arch 8 -model_dir se65
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 9 -model_dir se63
python -u transform_ckpt_to_pb.py -arch 9 -model_dir se63
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 10 -model_dir se35
python -u transform_ckpt_to_pb.py -arch 10 -model_dir se35
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 11 -model_dir se33
python -u transform_ckpt_to_pb.py -arch 11 -model_dir se33
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 12 -model_dir no65
python -u transform_ckpt_to_pb.py -arch 12 -model_dir no65
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 13 -model_dir no63
python -u transform_ckpt_to_pb.py -arch 13 -model_dir no63
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 14 -model_dir no35
python -u transform_ckpt_to_pb.py -arch 14 -model_dir no35
CUDA_VISIBLE_DEVICES=0 \
python -u train.py -arch 15 -model_dir no33
python -u transform_ckpt_to_pb.py -arch 15 -model_dir no33
