#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python train.py \
    "--mode" "train" \
    "--data_path" "/home/sy/dataset/UDD/UDD5" \
    "--train_batch_size" "16" \
    "--resume" "/home/sy/AerialSeg/DeepLab_UDD5_epoch10.pth.tar" \
    "--cuda" "True" \
    "--model" "DeepLabV3+" \
    "--dataset" "UDD5" \
    "--epochs" "100" \
    "--schedule_mode" "miou"