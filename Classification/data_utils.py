import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import logging
from datasets import load_dataset

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_rt_dataset(data_dir, model_name='distilbert-base-uncased'):
    """加载Rotten Tomatoes数据集（二分类）"""
    logger.info("加载RT数据集...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        # 尝试从本地加载
        if os.path.exists(os.path.join(data_dir, 'rt-train.pt')):
            logger.info("从本地加载RT数据...")
            train_dataset = torch.load(os.path.join(data_dir, 'rt-train.pt'))
            val_dataset = torch.load(os.path.join(data_dir, 'rt-val.pt'))
            test_dataset = torch.load(os.path.join(data_dir, 'rt-test.pt'))
        else:
            # 从HuggingFace加载
            logger.info("从HuggingFace加载RT数据...")
            dataset = load_dataset('rotten_tomatoes')
            
            # 分割数据集
            train_dataset = dataset['train']
            val_dataset = dataset['validation']
            test_dataset = dataset['test']
            
            # 保存到本地
            os.makedirs(data_dir, exist_ok=True)
            torch.save(train_dataset, os.path.join(data_dir, 'rt-train.pt'))
            torch.save(val_dataset, os.path.join(data_dir, 'rt-val.pt'))
            torch.save(test_dataset, os.path.join(data_dir, 'rt-test.pt'))
    except Exception as e:
        logger.error(f"加载RT数据集时出错: {e}")
        raise
    
    logger.info(f"RT数据集加载完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本, 测试集 {len(test_dataset)} 样本")
    return train_dataset, val_dataset, test_dataset, tokenizer

def load_sst5_dataset(data_dir, model_name='distilbert-base-uncased'):
    """加载SST-5数据集（五分类）"""
    logger.info("加载SST-5数据集...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        # 尝试从本地加载
        if os.path.exists(os.path.join(data_dir, 'sst5-train.pt')):
            logger.info("从本地加载SST-5数据...")
            train_dataset = torch.load(os.path.join(data_dir, 'sst5-train.pt'))
            val_dataset = torch.load(os.path.join(data_dir, 'sst5-val.pt'))
            test_dataset = torch.load(os.path.join(data_dir, 'sst5-test.pt'))
        else:
            # 从HuggingFace加载
            logger.info("从HuggingFace加载SST-5数据...")
            dataset = load_dataset('SetFit/sst5')
            
            # 分割数据集
            train_dataset = dataset['train']
            val_dataset = dataset['validation']
            test_dataset = dataset['test']
            
            # 保存到本地
            os.makedirs(data_dir, exist_ok=True)
            torch.save(train_dataset, os.path.join(data_dir, 'sst5-train.pt'))
            torch.save(val_dataset, os.path.join(data_dir, 'sst5-val.pt'))
            torch.save(test_dataset, os.path.join(data_dir, 'sst5-test.pt'))
    except Exception as e:
        logger.error(f"加载SST-5数据集时出错: {e}")
        raise
    
    logger.info(f"SST-5数据集加载完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本, 测试集 {len(test_dataset)} 样本")
    return train_dataset, val_dataset, test_dataset, tokenizer

def load_amazon_polarity(data_dir, model_name='distilbert-base-uncased'):
    """加载Amazon Polarity数据集（二分类，3.6M训练，400K测试）"""
    logger.info("加载Amazon Polarity数据集...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        # 尝试从本地加载
        if os.path.exists(os.path.join(data_dir, 'amazon_polarity-train.pt')):
            logger.info("从本地加载Amazon Polarity数据...")
            train_dataset = torch.load(os.path.join(data_dir, 'amazon_polarity-train.pt'))
            val_dataset = torch.load(os.path.join(data_dir, 'amazon_polarity-val.pt'))
            test_dataset = torch.load(os.path.join(data_dir, 'amazon_polarity-test.pt'))
        else:
            # 从HuggingFace加载
            logger.info("从HuggingFace加载Amazon Polarity数据...")
            dataset = load_dataset('amazon_polarity')
            
            # Amazon Polarity只有训练集和测试集，需要从训练集分出一部分作为验证集
            train_data = dataset['train']
            test_data = dataset['test']
            
            # 随机划分训练集，取10%作为验证集
            train_val_split = train_data.train_test_split(test_size=0.1)
            train_dataset = train_val_split['train']
            val_dataset = train_val_split['test']
            test_dataset = test_data
            
            # 转换标签字段，使其与其他数据集一致
            def convert_labels(example):
                # Amazon Polarity中，标签为0表示负面，1表示正面
                return {'labels': example['label'], 'text': example['content']}
            
            train_dataset = train_dataset.map(convert_labels)
            val_dataset = val_dataset.map(convert_labels)
            test_dataset = test_dataset.map(convert_labels)
            
            # 保存到本地
            os.makedirs(data_dir, exist_ok=True)
            torch.save(train_dataset, os.path.join(data_dir, 'amazon_polarity-train.pt'))
            torch.save(val_dataset, os.path.join(data_dir, 'amazon_polarity-val.pt'))
            torch.save(test_dataset, os.path.join(data_dir, 'amazon_polarity-test.pt'))
    except Exception as e:
        logger.error(f"加载Amazon Polarity数据集时出错: {e}")
        raise
    
    logger.info(f"Amazon Polarity数据集加载完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本, 测试集 {len(test_dataset)} 样本")
    return train_dataset, val_dataset, test_dataset, tokenizer

def create_dynamic_batch_sampler(dataset, tokens_per_batch=4096, max_length=128):
    """
    创建动态批次采样器，使每批次包含大约相同数量的token
    """
    # 一个简单的实现方式，假设所有样本长度相近
    approx_batch_size = max(1, tokens_per_batch // max_length)
    return approx_batch_size

def get_dataloaders(train_dataset, val_dataset, test_dataset, tokenizer=None, batch_size=32, max_length=128, tokens_per_batch=None):
    """创建数据加载器"""
    if tokenizer is None:
        # 如果没有传入tokenizer，假设dataset已经是TextClassificationDataset类型
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    else:
        # 创建Dataset
        train_ds = SentimentDataset(train_dataset, train_dataset['labels'], tokenizer, max_length)
        val_ds = SentimentDataset(val_dataset, val_dataset['labels'], tokenizer, max_length)
        test_ds = SentimentDataset(test_dataset, test_dataset['labels'], tokenizer, max_length)
        
        # 使用动态批处理
        if tokens_per_batch is not None:
            # 估计每个样本的token数量为max_length/2
            approx_tokens_per_sample = max_length / 2
            batch_size = max(1, int(tokens_per_batch / approx_tokens_per_sample))
            logger.info(f"使用近似批大小: {batch_size}，基于每批{tokens_per_batch}个tokens")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader 