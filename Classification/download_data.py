import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import logging
import pytreebank
import os
import shutil
import tarfile
import io
import ssl
import time
import random
import requests
from pathlib import Path

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def download_with_retry(url, output_file, max_retries=3):
    """带重试的下载函数"""
    for attempt in range(max_retries):
        try:
            # 创建SSL上下文，忽略证书验证
            context = ssl._create_unverified_context()
            
            # 添加User-Agent头，模拟浏览器请求
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=context, timeout=30) as response:
                with open(output_file, 'wb') as out_file:
                    out_file.write(response.read())
            
            logger.info(f"成功下载 {output_file}")
            return True
        except Exception as e:
            logger.warning(f"下载失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # 指数退避
                logger.info(f"等待 {sleep_time} 秒后重试...")
                time.sleep(sleep_time)
    
    logger.error(f"下载失败，已达到最大重试次数: {url}")
    return False

def download_rt_dataset(data_dir):
    """下载并处理Pang和Lee (2005)电影评论二分类数据集"""
    os.makedirs(data_dir, exist_ok=True)
    
    # 检查数据是否已存在
    if os.path.exists(os.path.join(data_dir, "rt_processed.csv")):
        logger.info("RT数据集已存在，跳过下载步骤")
        return
    
    rt_url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
    logger.info(f"从 {rt_url} 下载RT数据集...")
    
    # 下载数据
    try:
        response = urllib.request.urlopen(rt_url)
        tar_content = response.read()
        
        # 解压数据
        with tarfile.open(fileobj=io.BytesIO(tar_content), mode="r:gz") as tar:
            # 提取正面和负面评论文件
            for member in tar.getmembers():
                if member.name.endswith('.pos') or member.name.endswith('.neg'):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=data_dir)
    except Exception as e:
        logger.error(f"下载或解压RT数据集失败: {e}")
        # 尝试直接从备用源下载处理好的数据
        create_rt_fallback_data(data_dir)
        return
    
    # 检查文件是否存在，支持多种可能的文件名
    possible_pos_files = [
        os.path.join(data_dir, "rt-polaritydata.pos"),
        os.path.join(data_dir, "rt-polarity.pos"), 
        os.path.join(data_dir, "pos")
    ]
    
    possible_neg_files = [
        os.path.join(data_dir, "rt-polaritydata.neg"),
        os.path.join(data_dir, "rt-polarity.neg"),
        os.path.join(data_dir, "neg")
    ]
    
    # 找到实际存在的文件
    pos_file = None
    for file in possible_pos_files:
        if os.path.exists(file):
            pos_file = file
            break
    
    neg_file = None
    for file in possible_neg_files:
        if os.path.exists(file):
            neg_file = file
            break
    
    # 读取数据
    pos_texts = []
    neg_texts = []
    
    try:
        # 尝试读取文件
        if pos_file and neg_file:
            with open(pos_file, 'r', encoding='utf-8', errors='ignore') as f:
                pos_texts = [line.strip() for line in f]
            with open(neg_file, 'r', encoding='utf-8', errors='ignore') as f:
                neg_texts = [line.strip() for line in f]
        else:
            # 如果找不到文件，检查目录中的所有文件
            logger.error("找不到RT数据集文件")
            logger.info("目录中的文件:")
            for file in os.listdir(data_dir):
                logger.info(f" - {file}")
            
            # 尝试查找任何.pos和.neg文件
            pos_files = [f for f in os.listdir(data_dir) if f.endswith('.pos')]
            neg_files = [f for f in os.listdir(data_dir) if f.endswith('.neg')]
            
            if pos_files and neg_files:
                with open(os.path.join(data_dir, pos_files[0]), 'r', encoding='utf-8', errors='ignore') as f:
                    pos_texts = [line.strip() for line in f]
                with open(os.path.join(data_dir, neg_files[0]), 'r', encoding='utf-8', errors='ignore') as f:
                    neg_texts = [line.strip() for line in f]
            else:
                raise Exception("无法找到RT数据集的.pos和.neg文件")
    except Exception as e:
        logger.error(f"读取RT数据集失败: {e}")
        create_rt_fallback_data(data_dir)
        return
    
    # 合并数据并添加标签
    texts = pos_texts + neg_texts
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    
    # 保存处理后的数据
    df = pd.DataFrame({'text': texts, 'label': labels})
    processed_file = os.path.join(data_dir, "rt_processed.csv")
    df.to_csv(processed_file, index=False)
    
    logger.info(f"RT数据集处理完成，共 {len(texts)} 个样本，已保存到 {processed_file}")

def create_rt_fallback_data(data_dir):
    """创建RT数据集的备用数据（如果下载失败）"""
    logger.warning("创建RT数据集的备用数据...")
    
    # 使用huggingface datasets库下载数据
    try:
        from datasets import load_dataset
        
        # 加载rotten_tomatoes数据集作为替代
        logger.info("尝试从Hugging Face下载rotten_tomatoes数据集...")
        dataset = load_dataset("rotten_tomatoes")
        
        # 处理训练集和测试集
        train_texts = dataset["train"]["text"]
        train_labels = dataset["train"]["label"]
        test_texts = dataset["test"]["text"]
        test_labels = dataset["test"]["label"]
        
        # 合并数据
        texts = train_texts + test_texts
        labels = train_labels + test_labels
        
        logger.info(f"成功加载rotten_tomatoes数据集，共{len(texts)}个样本")
    except Exception as e:
        logger.error(f"加载rotten_tomatoes数据集失败: {e}")
        logger.warning("创建随机样本数据...")
        
        # 如果huggingface也失败，创建随机数据
        # 创建一些示例句子
        positive_templates = [
            "I really enjoyed this movie. {}", 
            "This film was excellent. {}", 
            "A wonderful cinematic experience. {}", 
            "Great acting and directing. {}", 
            "This movie is a masterpiece. {}"
        ]
        
        negative_templates = [
            "I disliked this movie. {}", 
            "This film was terrible. {}", 
            "A waste of time and money. {}", 
            "Poor acting and directing. {}", 
            "This movie is a disaster. {}"
        ]
        
        # 生成5000个正面和5000个负面评论
        pos_texts = []
        for i in range(5000):
            template = random.choice(positive_templates)
            pos_texts.append(template.format(f"Sample {i}"))
            
        neg_texts = []
        for i in range(5000):
            template = random.choice(negative_templates)
            neg_texts.append(template.format(f"Sample {i}"))
        
        texts = pos_texts + neg_texts
        labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    
    # 保存处理后的数据
    df = pd.DataFrame({'text': texts, 'label': labels})
    processed_file = os.path.join(data_dir, "rt_processed.csv")
    df.to_csv(processed_file, index=False)
    
    logger.warning(f"已创建RT数据集的备用数据，共 {len(texts)} 个样本，已保存到 {processed_file}")
    logger.warning("注意：这是替代数据，可能与原始RT数据集有所不同！")

def download_sst5_dataset(data_dir):
    """下载并处理Stanford Sentiment Treebank五分类数据集"""
    os.makedirs(data_dir, exist_ok=True)
    
    # 检查数据是否已存在
    if (os.path.exists(os.path.join(data_dir, "sst_train.csv")) and
        os.path.exists(os.path.join(data_dir, "sst_dev.csv")) and
        os.path.exists(os.path.join(data_dir, "sst_test.csv"))):
        logger.info("SST5数据集已存在，跳过下载步骤")
        return
    
    # 方法1: 使用pytreebank库下载(首选方法)
    try:
        logger.info("尝试使用pytreebank库下载SST5数据集...")
        import pytreebank
        
        # 检查是否安装了pytreebank
        logger.info("检查pytreebank版本...")
        
        # 下载数据集
        dataset = pytreebank.load_sst(data_dir)
        
        logger.info("pytreebank数据集已下载，准备处理为CSV格式...")
        
        # 准备数据
        splits = ["train", "dev", "test"]
        for split in splits:
            data = []
            for tree in dataset[split]:
                # SST-5标签: 0-4(非常负面到非常正面)
                for label, sentence in tree.to_labeled_lines():
                    data.append({"sentence": sentence, "label": label})
            
            # 将数据保存为CSV
            df = pd.DataFrame(data)
            output_file = os.path.join(data_dir, f"sst_{split}.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"SST5 {split}集: {len(df)} 个样本，保存到 {output_file}")
        
        logger.info("SST5数据集处理完成")
        return
    except Exception as e:
        logger.error(f"使用pytreebank下载SST5数据集失败: {str(e)}")
        logger.info("尝试备用下载方法...")
    
    # 方法2: 使用huggingface datasets
    try:
        from datasets import load_dataset
        
        logger.info("尝试从Hugging Face下载stanford_sentiment_treebank数据集...")
        dataset = load_dataset("sst")
        
        # 处理并保存训练集
        train_data = []
        for item in dataset["train"]:
            # 映射到五分类：(0-4) = 非常负面, 负面, 中性, 正面, 非常正面
            # 需要将原始标签(0-1浮点数)映射到0-4整数
            label = min(int(item["label"] * 5), 4)
            train_data.append({"sentence": item["sentence"], "label": label})
        
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(os.path.join(data_dir, "sst_train.csv"), index=False)
        logger.info(f"SST5训练集: {len(train_df)} 个样本")
        
        # 处理并保存验证集
        dev_data = []
        for item in dataset["validation"]:
            label = min(int(item["label"] * 5), 4)
            dev_data.append({"sentence": item["sentence"], "label": label})
        
        dev_df = pd.DataFrame(dev_data)
        dev_df.to_csv(os.path.join(data_dir, "sst_dev.csv"), index=False)
        logger.info(f"SST5验证集: {len(dev_df)} 个样本")
        
        # 处理并保存测试集
        test_data = []
        for item in dataset["test"]:
            label = min(int(item["label"] * 5), 4)
            test_data.append({"sentence": item["sentence"], "label": label})
        
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(os.path.join(data_dir, "sst_test.csv"), index=False)
        logger.info(f"SST5测试集: {len(test_df)} 个样本")
        
        logger.info("SST5数据集处理完成")
        return
    except Exception as e:
        logger.error(f"从Hugging Face下载SST数据集失败: {e}")
    
    # 方法3: 直接从Github下载CSV文件
    logger.info("尝试从Github下载SST5数据集CSV文件...")
    
    # 备用下载链接
    base_urls = [
        "https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data",
        "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/sst"
    ]
    
    success = False
    for base_url in base_urls:
        logger.info(f"尝试从 {base_url} 下载...")
        success = True
        
        # 下载训练集、验证集和测试集
        for split in ['train', 'dev', 'test']:
            url = f"{base_url}/sst_{split}.csv"
            output_file = os.path.join(data_dir, f"sst_{split}.csv")
            
            if not download_with_retry(url, output_file):
                success = False
                break
            
            # 验证文件内容
            try:
                df = pd.read_csv(output_file)
                logger.info(f"SST5 {split}集: {len(df)} 个样本")
            except Exception as e:
                logger.error(f"验证SST5 {split}集时出错: {e}")
                success = False
                break
        
        if success:
            logger.info("SST5数据集处理完成")
            return
    
    # 方法4: 下载原始数据集并处理
    try:
        logger.info("尝试下载原始SST数据集...")
        stanford_nlp_url = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"
        zip_path = os.path.join(data_dir, "sst.zip")
        
        if download_with_retry(stanford_nlp_url, zip_path):
            logger.info("解压原始SST数据集...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # 处理并转换为所需格式
            sst_dir = os.path.join(data_dir, "trees")
            if os.path.exists(sst_dir):
                process_stanford_sst(sst_dir, data_dir)
                logger.info("SST5数据集处理完成")
                return
    except Exception as e:
        logger.error(f"下载和处理原始SST数据集失败: {e}")
    
    # 如果所有下载尝试都失败，创建占位数据
    logger.warning("所有下载尝试都失败，创建SST5占位数据...")
    create_sst5_placeholder_data(data_dir)

def process_stanford_sst(sst_dir, output_dir):
    """处理斯坦福情感树库原始数据"""
    logger.info("处理原始Stanford Sentiment Treebank数据...")
    
    # 映射文件名到分割集
    split_files = {
        "train": os.path.join(sst_dir, "train.txt"),
        "dev": os.path.join(sst_dir, "dev.txt"),
        "test": os.path.join(sst_dir, "test.txt")
    }
    
    for split, file_path in split_files.items():
        if os.path.exists(file_path):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # 解析括号格式，例如 "(3 (2 It) (4 's) (3 good))"
                        # 这里使用简单的方法提取根标签和句子文本
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            label_str, rest = parts
                            # 提取标签 (去除左括号)
                            label = int(label_str.lstrip('('))
                            
                            # 提取句子 (去除标签和括号)
                            sentence = ' '.join(rest.replace('(', ' ').replace(')', ' ').split())
                            
                            # 映射标签 (原始SST使用0-4整数)
                            data.append({"sentence": sentence, "label": label})
                    except Exception as e:
                        logger.warning(f"解析行时出错: {line.strip()} - {e}")
                        continue
            
            # 保存到CSV
            df = pd.DataFrame(data)
            output_file = os.path.join(output_dir, f"sst_{split}.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"SST5 {split}集: {len(df)} 个样本")
    
def create_sst5_placeholder_data(data_dir):
    """创建SST5数据集的占位数据"""
    # 生成更真实的样本句子
    sentiment_templates = {
        0: [  # 非常负面
            "This movie was absolutely terrible. {}",
            "I hated every minute of this film. {}",
            "One of the worst experiences I've ever had. {}",
            "Completely unwatchable garbage. {}",
            "Awful in every possible way. {}"
        ],
        1: [  # 负面
            "I didn't enjoy this movie. {}",
            "Below average and disappointing. {}",
            "Not worth your time or money. {}",
            "Several problems made this film unenjoyable. {}",
            "I wouldn't recommend this to anyone. {}"
        ],
        2: [  # 中性
            "This film was just okay. {}",
            "Neither good nor bad, just average. {}",
            "Had some good parts and some bad parts. {}",
            "Watchable but forgettable. {}",
            "Mediocre at best. {}"
        ],
        3: [  # 正面
            "I enjoyed watching this movie. {}",
            "A good film with some memorable moments. {}",
            "Worth seeing at least once. {}",
            "Generally entertaining and well-made. {}",
            "Above average and satisfying. {}"
        ],
        4: [  # 非常正面
            "This is an excellent movie! {}",
            "One of the best films I've seen recently. {}",
            "Absolutely brilliant in every way. {}",
            "A true masterpiece of cinema. {}",
            "I loved everything about this film. {}"
        ]
    }
    
    # 生成数据集
    for split, size in [('train', 8544), ('dev', 1101), ('test', 2210)]:
        logger.info(f"创建SST5 {split}集占位数据...")
        
        sentences = []
        labels = []
        
        # 均匀分布的标签
        for i in range(size):
            label = i % 5
            template = random.choice(sentiment_templates[label])
            sentence = template.format(f"Sample {i} from {split} set.")
            
            sentences.append(sentence)
            labels.append(label)
        
        # 打乱数据
        combined = list(zip(sentences, labels))
        random.shuffle(combined)
        sentences, labels = zip(*combined)
        
        # 保存到CSV
        df = pd.DataFrame({'sentence': sentences, 'label': labels})
        output_file = os.path.join(data_dir, f"sst_{split}.csv")
        df.to_csv(output_file, index=False)
        
        logger.warning(f"已创建占位数据 {output_file}，包含 {size} 个样本")
    
    logger.warning("注意：这是占位数据，不代表真实的SST5数据集！")

def download_sst5_new():


    # 下载SST数据集
    dataset = pytreebank.load_sst("data/")

    # 准备训练/验证/测试数据
    for split in ["train", "dev", "test"]:
        data = []
        for tree in dataset[split]:
            # 提取句子和标签 (0-4: 非常负面到非常正面)
            for label, sentence in tree.to_labeled_lines():
                data.append({"sentence": sentence, "label": label})

        # 保存为CSV
        df = pd.DataFrame(data)
        df.to_csv(os.path.join("data", f"sst_{split}.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="下载并处理情感分析数据集")
    parser.add_argument("--data_dir", type=str, default="data", help="数据保存目录")
    parser.add_argument("--dataset", type=str, default="all", choices=["rt", "sst5", "all"], help="要下载的数据集")
    
    args = parser.parse_args()
    
    if args.dataset in ["rt", "all"]:
        download_rt_dataset(args.data_dir)
    
    if args.dataset in ["sst5", "all"]:
        download_sst5_new()
        # download_sst5_dataset(args.data_dir)
    
    logger.info("数据下载和处理完成！")
    logger.info(f"数据保存在目录: {os.path.abspath(args.data_dir)}")

if __name__ == "__main__":
    main() 