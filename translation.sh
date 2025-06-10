export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="3"

SEED=42
export HF_MIRROR="https://mirror.huggingface.com"

# 创建必要的目录
mkdir -p ./results/vit
mkdir -p ./results/translation
mkdir -p ./results/classification


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
