import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb

from data_utils import load_rt_dataset, load_sst5_dataset, get_dataloaders
from model import SentimentClassifier

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def evaluate_model(model, dataloader, criterion, device):
    """评估模型并返回详细指标"""
    model.eval()
    total_loss = 0
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
            
            # 记录损失和预测结果
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # 详细分类报告
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }

def plot_confusion_matrix(cm, class_names, output_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(args):
    # 初始化wandb (如果启用)
    if args.use_wandb:
        run_name = f"{args.dataset}-eval-{os.path.basename(args.model_path)}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "dataset": args.dataset,
                "model_path": args.model_path,
                "evaluation_type": "test"
            }
        )
        logger.info(f"Wandb初始化完成: {run_name}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据集
    logger.info(f"加载{args.dataset}数据集...")
    if args.dataset == 'rt':
        _, _, test_dataset, tokenizer = load_rt_dataset(args.data_dir)
        num_classes = 2
        class_names = ['负面', '正面']
    elif args.dataset == 'sst5':
        _, _, test_dataset, tokenizer = load_sst5_dataset(args.data_dir)
        num_classes = 5
        class_names = ['非常负面', '负面', '中性', '正面', '非常正面']
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 创建数据加载器
    _, _, test_loader = get_dataloaders(
        None, None, test_dataset, 
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
        max_len=args.max_length
    ).to(device)
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 评估模型
    logger.info("开始评估...")
    metrics = evaluate_model(model, test_loader, criterion, device)
    
    # 输出结果
    logger.info(f"测试集评估指标:")
    logger.info(f"Loss: {metrics['loss']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro-F1: {metrics['macro_f1']:.4f}")
    
    # 保存混淆矩阵
    os.makedirs(args.output_dir, exist_ok=True)
    cm_path = os.path.join(args.output_dir, f"{args.dataset}_confusion_matrix.png")
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
    logger.info(f"混淆矩阵已保存至: {cm_path}")
    
    # 保存详细指标到CSV
    metrics_file = os.path.join(args.output_dir, f"{args.dataset}_metrics.csv")
    metrics_df = pd.DataFrame(metrics['classification_report']).transpose()
    metrics_df.to_csv(metrics_file)
    logger.info(f"详细指标已保存至: {metrics_file}")
    
    # 输出每个类别的性能指标
    for i, class_name in enumerate(class_names):
        if str(i) in metrics['classification_report']:
            precision = metrics['classification_report'][str(i)]['precision']
            recall = metrics['classification_report'][str(i)]['recall']
            f1 = metrics['classification_report'][str(i)]['f1-score']
            support = metrics['classification_report'][str(i)]['support']
            logger.info(f"{class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}")
    
    # 记录wandb指标（如果启用）
    if args.use_wandb:
        # 记录主要指标
        wandb.log({
            "test/loss": metrics['loss'],
            "test/accuracy": metrics['accuracy'],
            "test/macro_f1": metrics['macro_f1'],
        })
        
        # 记录每个类别的指标
        for i, class_name in enumerate(class_names):
            if str(i) in metrics['classification_report']:
                class_metrics = metrics['classification_report'][str(i)]
                wandb.log({
                    f"test/class_{class_name}/precision": class_metrics['precision'],
                    f"test/class_{class_name}/recall": class_metrics['recall'],
                    f"test/class_{class_name}/f1": class_metrics['f1-score'],
                    f"test/class_{class_name}/support": class_metrics['support']
                })
        
        # 上传混淆矩阵图像
        wandb.log({"confusion_matrix_plot": wandb.Image(cm_path)})
        
        # 使用wandb内置的混淆矩阵可视化
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=metrics['labels'],
            preds=metrics['predictions'],
            class_names=class_names
        )})
        
        # 完成wandb会话
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="情感分类评估脚本")
    
    # 数据参数
    parser.add_argument("--dataset", type=str, default="rt", choices=["rt", "sst5"], help="数据集名称")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--tokens_per_batch", type=int, default=4096, help="每批次的token数量")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--d_model", type=int, default=128, help="模型维度")
    parser.add_argument("--nhead", type=int, default=8, help="注意力头数")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="前馈网络维度")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout率")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    
    # Wandb参数
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="Transformer-Text-Classification", help="Wandb项目名称")
    
    args = parser.parse_args()
    
    main(args) 