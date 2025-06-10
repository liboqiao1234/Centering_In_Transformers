export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="2"

SEED=42
export HF_MIRROR="https://mirror.huggingface.com"

# 创建必要的目录
mkdir -p ./results/vit
mkdir -p ./results/translation
mkdir -p ./results/classification



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
