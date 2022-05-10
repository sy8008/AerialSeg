#!/bin/bash

# "--finetune" "/home/sy/AerialSeg/checkpoint/ENet.pth" \
CUDA_VISIBLE_DEVICES=2 python train.py \
    "--mode" "train" \
    "--data_path" "/home/sy/dataset/uavid_processed" \
    "--train_batch_size" "16" \
    "--cuda" "True" \
    "--model" "DeepLabV3+" \
    "--dataset" "uavid" \
    "--epochs" "100" \
    "--schedule_mode" "step" \
    "--loss" "LS" 