import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
import logging
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from Taiyi.taiyi.monitor import Monitor
import wandb
from Taiyi.visualize import Visualization

from data_utils import load_rt_dataset, load_sst5_dataset, load_amazon_polarity, load_yahoo_answers, get_dataloaders
from model import SentimentClassifier

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
monitor = None
step = 0
vis_wandb = None


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 线性学习率调度器，包含预热期和线性衰减
class LinearWarmupLinearDecayLR:
    def __init__(self, optimizer, warmup_steps, total_steps, last_step=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.last_step = last_step
        self._step_count = 0
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            # 线性预热
            scale = max(1e-8, self._step_count / self.warmup_steps)
            return [group['lr'] * scale for group in self.optimizer.param_groups]
        else:
            # 线性衰减，但最低不低于原始学习率的1%
            progress = (self._step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = max(0.01, 1.0 - progress)  # 不低于1%
            return [group['lr'] * scale for group in self.optimizer.param_groups]
    
    def step(self):
        self._step_count += 1
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        return self._last_lr

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    global monitor, step, vis_wandb
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        # 更新学习率（如果使用步数级调度器）
        if scheduler is not None:
            scheduler.step()
            if args.use_wandb:
                current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
                wandb.log({"learning_rate": current_lr, "step": step})
        
        # 计算batch准确率
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        batch_labels = labels.cpu().numpy()
        batch_acc = accuracy_score(batch_labels, preds)
        batch_f1 = f1_score(batch_labels, preds, average='macro')
        
        # 使用wandb记录每个batch的指标
        if args.use_wandb:
            wandb.log({
                "batch_loss": loss.item(),
                "batch_acc": batch_acc,
                "batch_f1": batch_f1,
                "step": step
            })
            
            # 模型监控和可视化
            monitor.track(step)
            vis_wandb.show(step)
        
        step += 1
        
        # 记录损失和预测结果
        epoch_loss += loss.item()
        all_preds.extend(preds)
        all_labels.extend(batch_labels)
    
    # 计算指标
    avg_loss = epoch_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, macro_f1

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # 计算batch准确率
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_labels = labels.cpu().numpy()
            batch_acc = accuracy_score(batch_labels, preds)
            batch_f1 = f1_score(batch_labels, preds, average='macro')
            
            # 使用wandb记录每个验证batch的指标
            if args.use_wandb:
                wandb.log({
                    "val_batch_loss": loss.item(),
                    "val_batch_acc": batch_acc,
                    "val_batch_f1": batch_f1
                })
            
            # 记录损失和预测结果
            epoch_loss += loss.item()
            all_preds.extend(preds)
            all_labels.extend(batch_labels)
    
    # 计算指标
    avg_loss = epoch_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, macro_f1

def save_checkpoint(model, optimizer, epoch, best_val_f1, save_path):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1': best_val_f1
    }, save_path)
    logger.info(f"模型保存到: {save_path}")

def main(args):
    taiyi_config = {
        "LayerNorm": [['InputAngleStd','linear(5,0)'], ['InputAngleMean', 'linear(5,0)'],
                        ['OutputAngleStd','linear(5,0)'], ['OutputAngleMean', 'linear(5,0)']],
        "RMSNorm": [['InputAngleStd','linear(5,0)'], ['InputAngleMean', 'linear(5,0)'],
                    ['OutputAngleStd','linear(5,0)'], ['OutputAngleMean', 'linear(5,0)']],
        "MultiheadAttention": [['InputAngleStd','linear(5,0)'], ['InputAngleMean', 'linear(5,0)'],
                        ['OutputAngleStd','linear(5,0)'], ['OutputAngleMean', 'linear(5,0)']],
        "Mlp": [['InputAngleStd','linear(5,0)'], ['InputAngleMean', 'linear(5,0)'],
                        ['OutputAngleStd','linear(5,0)'], ['OutputAngleMean', 'linear(5,0)']],
    }
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据集
    if args.dataset == 'yahoo_answers':
        logger.info("加载Yahoo! Answers数据集...")
        train_dataset, val_dataset, test_dataset, tokenizer = load_yahoo_answers(args.data_dir)
        num_classes = 10
        class_names = ['社会与文化', '科学与数学', '健康', '教育与参考', '计算机与互联网',
                      '体育', '商业与金融', '娱乐与音乐', '家庭与人际关系', '政治与政府']
    elif args.dataset == 'amazon_polarity':
        logger.info("加载Amazon Polarity数据集...")
        train_dataset, val_dataset, test_dataset, tokenizer = load_amazon_polarity(args.data_dir)
        num_classes = 2
        class_names = ['负面', '正面']
    elif args.dataset == 'sst5':
        logger.info("加载SST-5数据集...")
        train_dataset, val_dataset, test_dataset, tokenizer = load_sst5_dataset(args.data_dir)
        num_classes = 5
        class_names = ['非常负面', '负面', '中性', '正面', '非常正面']
    elif args.dataset == 'rt':
        logger.info("加载Rotten Tomatoes数据集...")
        train_dataset, val_dataset, test_dataset, tokenizer = load_rt_dataset(args.data_dir)
        num_classes = 2
        class_names = ['负面', '正面']
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, 
        tokenizer=tokenizer,
        tokens_per_batch=args.tokens_per_batch,
        max_length=args.max_length
    )
    
    # 创建模型
    logger.info("初始化模型...")
    vocab_size = tokenizer.vocab_size
    model = SentimentClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.max_length,
        norm_type=args.norm_type
    ).to(device)
    
    # 记录模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数: {total_params:,} 可训练参数: {trainable_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(args.beta1, args.beta2), 
        eps=args.eps,
        weight_decay=0.01
    )
    
    # 设置学习率调度器 - 使用线性预热和线性衰减
    # 计算总训练步数和预热步数
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.05)  # 预热步数为总步数的5%
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_val_f1 = 0
    best_epoch = 0

    # 初始化wandb
    if args.use_wandb:
        norm_suffix = f"-{args.norm_type}"
        run_name = f"{args.dataset}-transformer-{args.d_model}d-{args.num_layers}l{norm_suffix}"
        wandb.init(
            project="text-classification",
            name=run_name,
            config={
                "dataset": args.dataset,
                "model": "transformer",
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "nhead": args.nhead,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "batch_size": args.tokens_per_batch,
                "max_length": args.max_length,
                "epochs": args.epochs,
                "lr": args.lr,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
                "norm_type": args.norm_type,
                "num_classes": num_classes
            }
        )
        
        global monitor, vis_wandb, step
        step = 0
        monitor = Monitor(model, taiyi_config)
        vis_wandb = Visualization(monitor, wandb)
    
    # 创建输出目录
    checkpoint_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练循环
    logger.info(f"开始训练，总训练轮数: {args.epochs}，预热步数: {warmup_steps}/{total_steps}")
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        
        # 验证
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        
        # 记录指标
        logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        if args.use_wandb:
            current_lr = scheduler._last_lr[0]
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "learning_rate": current_lr
            })
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            save_checkpoint(
                model, 
                optimizer, 
                epoch, 
                best_val_f1, 
                os.path.join(checkpoint_dir, 'best_model.pt')
            )
    
    # 使用最佳模型进行测试
    logger.info(f"加载最佳模型 (epoch {best_epoch+1})...")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    logger.info(f"测试结果 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    if args.use_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1
        })
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="文本分类模型训练")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--dataset", type=str, default="amazon_polarity", help="数据集名称: rt, sst5 或 amazon_polarity")
    parser.add_argument("--max_length", type=int, default=256, help="最大序列长度")
    parser.add_argument("--tokens_per_batch", type=int, default=4096, help="每个批次的大约token数")
    
    # 模型参数
    parser.add_argument("--d_model", type=int, default=128, help="隐藏层大小")
    parser.add_argument("--nhead", type=int, default=8, help="注意力头数")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="FFN隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout率")
    parser.add_argument("--norm_type", type=str, default="layernorm", choices=["layernorm", "rmsnorm"], help="归一化类型")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录实验")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    main(args) 