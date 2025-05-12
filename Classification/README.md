# 情感分类实现

本项目实现了基于Transformer编码器的情感分类模型，用于处理RT（二分类）和SST5（五分类）情感分析任务。

## 项目结构

- `data_utils.py`: 数据加载和预处理
- `model.py`: 模型架构定义
- `train.py`: 训练脚本
- `evaluate.py`: 评估脚本
- `inference.py`: 推理脚本
- `download_data.py`: 数据下载和准备脚本
- `run_sentiment_analysis.py`: 一键运行脚本

## 快速开始

最简单的方法是使用一键运行脚本，它将自动执行下载数据、训练和评估的全流程：

```bash
# 使用RT数据集（二分类）
python run_sentiment_analysis.py --dataset rt --epochs 10

# 使用SST5数据集（五分类）
python run_sentiment_analysis.py --dataset sst5 --epochs 15

# 仅评估已训练的模型
python run_sentiment_analysis.py --dataset rt --skip_download --skip_train

# 指定自定义目录
python run_sentiment_analysis.py --dataset rt --data_dir custom_data --output_dir custom_output

# 启用Weights & Biases日志记录
python run_sentiment_analysis.py --dataset rt --use_wandb --install_deps
```

## Weights & Biases 集成

本项目支持使用 [Weights & Biases](https://wandb.ai/) 进行实验跟踪和可视化。

### 启用 Wandb

要启用 Wandb 日志记录，需要以下步骤：

1. 安装 Wandb：`pip install wandb`
2. 登录 Wandb 账户：`wandb login` 或使用 API 密钥作为参数
3. 在运行脚本时添加 `--use_wandb` 参数

```bash
# 首次使用时安装依赖并登录 Wandb
python run_sentiment_analysis.py --dataset rt --use_wandb --install_deps

# 使用 API 密钥直接登录
python run_sentiment_analysis.py --dataset rt --use_wandb --wandb_login YOUR_API_KEY

# 设置自定义项目名称
python run_sentiment_analysis.py --dataset rt --use_wandb --wandb_project "My-Text-Classification"
```

### 跟踪指标

Wandb 将自动记录以下指标：

- 训练和验证的损失、准确率和 F1 分数
- 学习率变化
- 测试集性能指标
- 混淆矩阵可视化
- 每个类别的精确率、召回率和 F1 分数

你可以在 Wandb 网站上查看这些指标的实时更新和可视化。

## 环境要求

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm wandb
```

## 数据准备

### 自动下载数据

最简单的方法是使用提供的下载脚本：

```bash
# 下载所有数据集
python download_data.py

# 仅下载RT数据集
python download_data.py --dataset rt

# 仅下载SST5数据集
python download_data.py --dataset sst5

# 指定数据保存目录
python download_data.py --data_dir custom_data_path
```

### 手动准备数据

如果需要手动准备数据，请按以下步骤操作：

#### RT数据集（二分类情感分析）

1. 下载Pang和Lee (2005)电影评论数据集：https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
2. 解压文件，将`rt-polaritydata.pos`和`rt-polaritydata.neg`放入`data/`目录

#### SST5数据集（五分类情感分析）

1. 下载处理好的SST5数据：
   - 训练集：https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst_train.csv
   - 验证集：https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst_dev.csv
   - 测试集：https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst_test.csv
2. 将这些文件放入`data/`目录

## 模型架构

- Embedding + 位置编码：128维向量 + 正弦/余弦位置编码
- 4层Transformer Encoder
  - 多头自注意力（8头，隐藏维度128）
  - 前馈网络（维度512）
  - Dropout=0.2
  - LayerNorm（Pre-Norm架构）
- 分类头：取第一个token的表示，映射到类别数（RT=2，SST5=5）

## 训练模型

```bash
# 首先确保数据已准备好
python download_data.py

# RT数据集训练
python train.py --dataset rt --data_dir data --output_dir outputs --epochs 15

# SST5数据集训练
python train.py --dataset sst5 --data_dir data --output_dir outputs --epochs 15

# 使用Wandb记录训练过程
python train.py --dataset rt --data_dir data --output_dir outputs --epochs 15 --use_wandb
```

### 主要参数说明

- `--dataset`: 数据集名称，'rt'或'sst5'
- `--data_dir`: 数据目录
- `--max_length`: 最大序列长度（默认128）
- `--tokens_per_batch`: 每批次的token数量（默认4096）
- `--d_model`: 模型维度（默认128）
- `--nhead`: 注意力头数（默认8）
- `--dim_feedforward`: 前馈网络维度（默认512）
- `--num_layers`: Transformer层数（默认4）
- `--dropout`: Dropout率（默认0.2）
- `--epochs`: 训练轮数（默认15）
- `--lr`: 学习率（默认1e-3）
- `--output_dir`: 输出目录（默认'outputs'）
- `--use_wandb`: 启用Wandb日志记录（默认关闭）
- `--wandb_project`: Wandb项目名称（默认'Transformer-Text-Classification'）

## 评估模型

```bash
# 评估RT模型
python evaluate.py --dataset rt --data_dir data --model_path outputs/rt_best.pt

# 评估SST5模型
python evaluate.py --dataset sst5 --data_dir data --model_path outputs/sst5_best.pt

# 使用Wandb记录评估指标
python evaluate.py --dataset rt --data_dir data --model_path outputs/rt_best.pt --use_wandb
```

评估结果将输出：
- 准确率（Accuracy）和宏平均F1分数（Macro-F1）
- 每个类别的精确率、召回率和F1分数
- 混淆矩阵图像保存至输出目录
- 详细评估指标保存至CSV文件

## 推理预测

训练好的模型可以用于对新文本进行情感分析：

### 交互式模式

```bash
# RT模型交互式预测
python inference.py --model_path outputs/rt_best.pt --dataset rt

# SST5模型交互式预测
python inference.py --model_path outputs/sst5_best.pt --dataset sst5
```

### 批量预测模式

```bash
# 对文本文件中的每行进行预测，并保存结果
python inference.py --model_path outputs/rt_best.pt --dataset rt --input_file texts.txt --output_file results.json
```

### 主要参数说明

- `--model_path`: 模型权重路径（必需）
- `--dataset`: 数据集类型（'rt'或'sst5'）
- `--tokenizer_name`: 使用的tokenizer名称（默认'bert-base-uncased'）
- `--input_file`: 输入文本文件（每行一个文本）
- `--output_file`: 输出结果JSON文件
- `--cpu`: 强制使用CPU进行推理

## 计算资源需求

- **显存需求**：最低150-200MB，可在大多数GPU上运行（包括入门级GPU）
- **训练时间**：
  - GPU上约5-15分钟（取决于GPU性能）
  - CPU上约30-75分钟
- **模型大小**：约20MB
- **推理速度**：单条文本在GPU上<10ms，CPU上<100ms

## 性能表现

在合理训练和调参后，模型预期性能：

- RT（二分类）: Accuracy > 80%，Macro-F1 > 80%
- SST5（五分类）: Accuracy > 45%，Macro-F1 > 45%

## 模型应用

训练好的模型可用于：
- 电影评论情感分析
- 客户评价情感分类
- 社交媒体情感监测 