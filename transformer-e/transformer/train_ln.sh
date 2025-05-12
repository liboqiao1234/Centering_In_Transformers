#!/bin/bash

# 使用LayerNorm训练模型
echo "开始使用LayerNorm训练模型..."
python train.py --seed 42 --norm-type LayerNorm --epochs 20 --batch-size 128 --d-model 512 --n-layers 6 --n-heads 8 --ffn-hidden 2048

# 训练完成后关机
echo "训练完成，系统将在1分钟后关机..."
sleep 60
sudo shutdown -h now 