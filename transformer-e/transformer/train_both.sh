#!/bin/bash

# 设置随机种子和其他超参数
SEED=1
EPOCHS=80
BATCH_SIZE=128
D_MODEL=512
N_LAYERS=6
N_HEADS=8
FFN_HIDDEN=2048
INIT_LR=0.0001
WARMUP=4000  # 对于transformer调度器，这是步数；对于ReduceLROnPlateau，这是轮数
PATIENCE=5
USE_TRANSFORMER_LR=true  # 使用Transformer原始学习率调度器

# 创建日志目录
LOG_DIR="train_logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/train_comparison_$(date +%Y%m%d_%H%M%S).log"

# 构建学习率调度器参数
if [ "$USE_TRANSFORMER_LR" = true ] ; then
    LR_SCHEDULER_ARGS="--use-transformer-lr"
    LR_SCHEDULER_TYPE="transformer_warmup"
else
    LR_SCHEDULER_ARGS=""
    LR_SCHEDULER_TYPE="reduce_on_plateau"
fi

# 记录基本信息到日志
echo "============================================================" | tee -a $LOG_FILE
echo "训练开始时间: $(date)" | tee -a $LOG_FILE
echo "随机种子: $SEED" | tee -a $LOG_FILE
echo "训练轮数: $EPOCHS" | tee -a $LOG_FILE
echo "批次大小: $BATCH_SIZE" | tee -a $LOG_FILE
echo "模型维度: $D_MODEL" | tee -a $LOG_FILE
echo "层数: $N_LAYERS" | tee -a $LOG_FILE
echo "注意力头数: $N_HEADS" | tee -a $LOG_FILE
echo "前馈网络隐藏层维度: $FFN_HIDDEN" | tee -a $LOG_FILE
echo "初始学习率: $INIT_LR" | tee -a $LOG_FILE
echo "学习率调度器: $LR_SCHEDULER_TYPE" | tee -a $LOG_FILE
if [ "$USE_TRANSFORMER_LR" = true ] ; then
    echo "预热步数: $WARMUP" | tee -a $LOG_FILE
else
    echo "预热轮数: $WARMUP" | tee -a $LOG_FILE
    echo "学习率调度器耐心值: $PATIENCE" | tee -a $LOG_FILE
fi
echo "============================================================" | tee -a $LOG_FILE

# 第一步：使用LayerNorm训练模型
echo "==========================================================" | tee -a $LOG_FILE
echo "开始使用LayerNorm训练模型..." | tee -a $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE
echo "==========================================================" | tee -a $LOG_FILE
# python train.py --seed $SEED --norm-type LayerNorm --epochs $EPOCHS --batch-size $BATCH_SIZE \
#               --d-model $D_MODEL --n-layers $N_LAYERS --n-heads $N_HEADS --ffn-hidden $FFN_HIDDEN \
#               --learning-rate $INIT_LR --warmup $WARMUP --patience $PATIENCE $LR_SCHEDULER_ARGS 2>&1 | tee -a $LOG_FILE

# 记录LayerNorm训练结果
echo "LayerNorm训练完成时间: $(date)" | tee -a $LOG_FILE
echo "保存最新的LayerNorm模型文件:" | tee -a $LOG_FILE
find saved -name "model-LayerNorm-seed${SEED}*" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2- | tee -a $LOG_FILE
echo "==========================================================" | tee -a $LOG_FILE

# 第二步：使用RMSNorm训练模型
echo "==========================================================" | tee -a $LOG_FILE
echo "开始使用RMSNorm训练模型..." | tee -a $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE
echo "==========================================================" | tee -a $LOG_FILE
python train.py --seed $SEED --norm-type RMS --epochs $EPOCHS --batch-size $BATCH_SIZE \
               --d-model $D_MODEL --n-layers $N_LAYERS --n-heads $N_HEADS --ffn-hidden $FFN_HIDDEN \
               --learning-rate $INIT_LR --warmup $WARMUP --patience $PATIENCE $LR_SCHEDULER_ARGS 2>&1 | tee -a $LOG_FILE

# 记录RMSNorm训练结果
echo "RMSNorm训练完成时间: $(date)" | tee -a $LOG_FILE
echo "保存最新的RMSNorm模型文件:" | tee -a $LOG_FILE
find saved -name "model-RMS-seed${SEED}*" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2- | tee -a $LOG_FILE
echo "==========================================================" | tee -a $LOG_FILE

# 总结训练结果
echo "所有训练任务已完成!" | tee -a $LOG_FILE
echo "完成时间: $(date)" | tee -a $LOG_FILE
echo "日志文件已保存到: $LOG_FILE" | tee -a $LOG_FILE
echo "训练模型已保存到saved目录" | tee -a $LOG_FILE
echo "==========================================================" | tee -a $LOG_FILE

# 创建学习率变化和性能对比总结
SUMMARY_FILE="$LOG_DIR/results_summary.txt"
echo "训练结果总结 - $(date)" > $SUMMARY_FILE
echo "============================================================" >> $SUMMARY_FILE
echo "训练参数:" >> $SUMMARY_FILE
echo "初始学习率: $INIT_LR" >> $SUMMARY_FILE
echo "学习率调度器: $LR_SCHEDULER_TYPE" >> $SUMMARY_FILE
if [ "$USE_TRANSFORMER_LR" = true ] ; then
    echo "预热步数: $WARMUP" >> $SUMMARY_FILE
else
    echo "预热轮数: $WARMUP" >> $SUMMARY_FILE
    echo "学习率调度器耐心值: $PATIENCE" >> $SUMMARY_FILE
fi
echo "============================================================" >> $SUMMARY_FILE
echo "LayerNorm 模型:" >> $SUMMARY_FILE
grep -A 3 "Test Loss" $LOG_FILE | grep -B 3 -A 0 "RMSNorm" | head -n 1 >> $SUMMARY_FILE
echo >> $SUMMARY_FILE
echo "LayerNorm 学习率变化:" >> $SUMMARY_FILE
grep "Learning Rate:" $LOG_FILE | head -n 10 >> $SUMMARY_FILE
echo "..." >> $SUMMARY_FILE
grep "Learning Rate:" $LOG_FILE | grep -A 10 -B 0 "RMSNorm" | head -n 5 >> $SUMMARY_FILE
echo "============================================================" >> $SUMMARY_FILE
echo "RMSNorm 模型:" >> $SUMMARY_FILE
grep -A 0 "Test Loss" $LOG_FILE | tail -n 1 >> $SUMMARY_FILE
echo >> $SUMMARY_FILE
echo "RMSNorm 学习率变化:" >> $SUMMARY_FILE
grep "Learning Rate:" $LOG_FILE | grep -A 10 "RMSNorm" | head -n 10 >> $SUMMARY_FILE
echo "..." >> $SUMMARY_FILE
grep "Learning Rate:" $LOG_FILE | tail -n 5 >> $SUMMARY_FILE
echo "============================================================" >> $SUMMARY_FILE
echo "完整日志: $LOG_FILE" >> $SUMMARY_FILE

echo "训练结果总结已保存到: $SUMMARY_FILE" | tee -a $LOG_FILE

# 训练完成后关机
# echo "==========================================================" | tee -a $LOG_FILE
# echo "所有训练任务已完成，系统将在1分钟后关机..." | tee -a $LOG_FILE
# echo "==========================================================" | tee -a $LOG_FILE
# sleep 60
# sudo shutdown -h now 
