CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/CSG/bagtricks_gvit.yml MODEL.DEVICE "cuda:0"
##!/bin/bash
#echo "Starting training..."
#CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/CSG/bagtricks_gvit.yml MODEL.DEVICE "cuda:0"
#echo "Training finished."

#python tools/train_net.py --config-file ./configs/CSG/bagtricks_gvit.yml MODEL.DEVICE "cuda:0"

'''从checkpoint恢复训练'''
#python tools/train_net.py --config-file ./configs/CSG/bagtricks_gvit.yml --resume MODEL.DEVICE "cuda:0"
