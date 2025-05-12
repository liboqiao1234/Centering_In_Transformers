import argparse
import torch
import torch.nn as nn
import os
import logging
from transformers import AutoTokenizer

from model import SentimentClassifier

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_model(model_path, vocab_size, num_classes, d_model=128, nhead=8, 
              dim_feedforward=512, num_layers=4, dropout=0.2, max_len=128, device='cuda'):
    """加载保存的模型"""
    model = SentimentClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        dropout=dropout,
        max_len=max_len
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def predict_sentiment(text, model, tokenizer, max_length=128, device='cuda'):
    """预测文本情感"""
    # 对输入文本进行编码
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 使用模型预测
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    return pred_class, confidence, probs[0].cpu().numpy()

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # 设置类别标签
    if args.dataset == 'rt':
        num_classes = 2
        class_names = ['负面', '正面']
    elif args.dataset == 'sst5':
        num_classes = 5
        class_names = ['非常负面', '负面', '中性', '正面', '非常正面']
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    model = load_model(
        args.model_path, 
        vocab_size=tokenizer.vocab_size,
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.max_length,
        device=device
    )
    
    # 处理输入文本
    if args.input_file:
        # 从文件读取多个文本
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        # 批量预测
        results = []
        for i, text in enumerate(texts):
            pred_class, confidence, probs = predict_sentiment(
                text, model, tokenizer, args.max_length, device
            )
            
            result = {
                'text': text,
                'predicted_class': pred_class,
                'predicted_label': class_names[pred_class],
                'confidence': confidence,
                'probabilities': {class_names[i]: float(probs[i]) for i in range(len(class_names))}
            }
            results.append(result)
            
            logger.info(f"文本 {i+1}: {text}")
            logger.info(f"预测结果: {class_names[pred_class]} (置信度: {confidence:.4f})")
            logger.info("各类别概率:")
            for i, class_name in enumerate(class_names):
                logger.info(f"  {class_name}: {probs[i]:.4f}")
            logger.info("-" * 50)
        
        # 保存结果到文件
        if args.output_file:
            import json
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存至: {args.output_file}")
    
    else:
        # 交互式模式
        logger.info("进入交互式预测模式（输入'exit'退出）")
        while True:
            text = input("\n请输入文本: ")
            if text.lower() == 'exit':
                break
            
            pred_class, confidence, probs = predict_sentiment(
                text, model, tokenizer, args.max_length, device
            )
            
            print(f"预测结果: {class_names[pred_class]} (置信度: {confidence:.4f})")
            print("各类别概率:")
            for i, class_name in enumerate(class_names):
                print(f"  {class_name}: {probs[i]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="情感分类推理脚本")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--dataset", type=str, default="rt", choices=["rt", "sst5"], help="数据集名称")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", help="Tokenizer名称")
    
    # 模型架构参数
    parser.add_argument("--d_model", type=int, default=128, help="模型维度")
    parser.add_argument("--nhead", type=int, default=8, help="注意力头数")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="前馈网络维度")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout率")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    
    # 输入输出参数
    parser.add_argument("--input_file", type=str, help="输入文本文件路径（每行一个文本，如果不提供则进入交互模式）")
    parser.add_argument("--output_file", type=str, help="输出结果文件路径（仅在提供input_file时有效）")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU进行推理")
    
    args = parser.parse_args()
    
    main(args) 