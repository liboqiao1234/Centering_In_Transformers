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