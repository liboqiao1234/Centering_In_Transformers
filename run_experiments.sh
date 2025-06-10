#!/bin/bash

# 设置通用参数
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="1"

SEED=42
export HF_MIRROR="https://mirror.huggingface.com"

# 创建必要的目录
mkdir -p ./results/vit
mkdir -p ./results/translation
mkdir -p ./results/classification

echo "运行统一实验，比较LayerNorm和RMSNorm效果"
echo "====================================="

# ViT on CIFAR-10 实验 (100 epochs)
echo "开始ViT on CIFAR-10实验"
echo "-----------------------------------"

# LayerNorm
echo "运行ViT with LayerNorm..."
cd cbwc-vit
python train.py \
  --norm_type ln \
  --data_path ./data \
  --dump_path ../results/vit \
  --epochs 100 \
  --batch_size 512 \
  --lr 1e-4 \
  --wd 0.1 \
  --warmup_epochs 5 \
  --patch_size 4 \
  --img_size 32\
  --dropout 0.1 \
  --num_classes 10 \
  --seed $SEED \
  --wandb True

# RMSNorm
echo "运行ViT with RMSNorm..."
python train.py \
  --norm_type rms \
  --data_path ./data \
  --dump_path ../results/vit \
  --epochs 100 \
  --batch_size 512 \
  --img_size 32\
  --lr 1e-4 \
  --wd 0.1 \
  --warmup_epochs 5 \
  --patch_size 4 \
  --dropout 0.1 \
  --num_classes 10 \
  --seed $SEED \
  --wandb True

cd ..

# 文本分类实验 (Yahoo! Answers, 10分类，数据集更小)
echo "开始文本分类 (Yahoo! Answers) 实验"
echo "-----------------------------------"

# LayerNorm
echo "运行文本分类 with LayerNorm..."
cd Classification
python train.py \
  --norm_type layernorm \
  --data_dir ./data \
  --output_dir ../results/classification \
  --dataset yahoo_answers \
  --epochs 20 \
  --lr 5e-4 \
  --dropout 0.1 \
  --tokens_per_batch 24576 \
  --max_length 160 \
  --d_model 320 \
  --nhead 10 \
  --dim_feedforward 1280 \
  --num_layers 6 \
  --seed $SEED \
  --use_wandb

# RMSNorm
echo "运行文本分类 with RMSNorm..."
python train.py \
  --norm_type rmsnorm \
  --data_dir ./data \
  --output_dir ../results/classification \
  --dataset yahoo_answers \
  --epochs 20 \
  --lr 5e-4 \
  --dropout 0.1 \
  --tokens_per_batch 24576 \
  --max_length 160 \
  --d_model 320 \
  --nhead 10 \
  --dim_feedforward 1280 \
  --num_layers 6 \
  --seed $SEED \
  --use_wandb

cd ..

# 机器翻译实验 (En->De, 减少到40 epochs)
echo "开始机器翻译 (En->De) 实验"
echo "-----------------------------------"

# LayerNorm
echo "运行机器翻译 with LayerNorm..."
cd transformer-e/transformer
python train.py \
  --norm-type ln \
  --epochs 40 \
  --batch-size 384 \
  --d-model 768 \
  --ffn-hidden 3072 \
  --learning-rate 5e-4 \
  --dropout 0.2 \
  --warmup-ratio 0.1 \
  --clip 1.0 \
  --seed $SEED

# RMSNorm
echo "运行机器翻译 with RMSNorm..."
python train.py \
  --norm-type rms \
  --epochs 40 \
  --batch-size 384 \
  --d-model 768 \
  --ffn-hidden 3072 \
  --learning-rate 5e-4 \
  --dropout 0.2 \
  --warmup-ratio 0.1 \
  --clip 1.0 \
  --seed $SEED

cd ../..

echo "====================================="
echo "所有实验完成!" 

# shutdown
