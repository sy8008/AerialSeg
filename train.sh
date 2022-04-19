#!/bin/bash

# "--finetune" "/home/sy/AerialSeg/checkpoint/ENet.pth" \
CUDA_VISIBLE_DEVICES=3 python train.py \
    "--mode" "train" \
    "--data_path" "/home/sy/dataset/ArsUDD" \
    "--train_batch_size" "8" \
    "--cuda" "True" \
    "--model" "DeepLabV3+" \
    "--dataset" "ArsUDD" \
    "--epochs" "100" \
    "--schedule_mode" "step" \
    "--loss" "LS"