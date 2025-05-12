# 文本分类批量实验系统

本文档提供了使用批处理脚本运行文本分类实验的说明。该系统允许您批量测试不同的数据集、归一化类型和随机种子组合，以分析它们对模型性能的影响。

## 主要特性

1. 支持两个数据集：RT（电影评论）和SST5（斯坦福情感树库）
2. 支持两种归一化类型：LayerNorm和RMSNorm
3. 可以使用多个随机种子进行实验，以评估模型稳定性
4. 提供结果分析工具，生成统计数据和可视化图表

## 脚本说明

系统包含以下三个主要脚本：

1. `run_experiments.bat` - 主控制脚本，提供菜单界面
2. `run_batch_experiments.bat` - 运行完整批量实验的脚本
3. `analyze_results.py` - 分析和可视化实验结果的Python脚本

## 使用方法

### 1. 运行实验菜单

直接在命令行中运行以下命令启动实验菜单：

```
cd Classification
run_experiments.bat
```

这将显示一个交互式菜单，您可以选择：
- 运行全部实验
- 仅运行特定数据集的实验
- 仅运行特定归一化类型的实验
- 仅使用特定随机种子运行实验
- 自定义实验参数
- 分析实验结果
- 退出

### 2. 直接运行完整批量实验

如果您想直接运行包含所有组合的完整批量实验，可以执行：

```
cd Classification
run_batch_experiments.bat
```

这将运行所有数据集（RT和SST5）、所有归一化类型（LayerNorm和RMSNorm）和所有预定义随机种子（42, 123, 456, 789, 1024）的组合，总共20个实验。

### 3. 分析实验结果

实验完成后，您可以通过以下命令分析结果：

```
cd Classification
python analyze_results.py
```

这将分析`experiment_results`目录中的结果文件，生成统计数据和可视化图表，包括：
- 各组合的准确率和F1分数汇总
- 不同数据集和归一化类型下准确率和F1分数的对比图
- 不同实验设置的性能分布箱线图

分析结果将保存在`experiment_results`目录中的以下文件：
- `analysis_results.xlsx` - 包含原始数据和统计结果的Excel文件
- `results_comparison.png` - 性能对比图
- `results_boxplot.png` - 性能分布箱线图

## 文件结构

实验完成后，您将获得以下文件结构：

```
Classification/
│
├── run_experiments.bat           # 主控制脚本
├── run_batch_experiments.bat     # 批量实验脚本
├── analyze_results.py            # 结果分析脚本
├── README_BATCH_EXPERIMENTS.md   # 本说明文档
│
├── experiment_results/           # 实验结果目录
│   ├── experiment_log.txt        # 实验运行日志
│   ├── custom_experiment_log.txt # 自定义实验日志
│   ├── rt_layernorm_seed42_results.txt  # 各实验的结果文件
│   ├── rt_rmsnorm_seed42_results.txt
│   ├── ...
│   ├── analysis_results.xlsx     # 实验结果Excel分析表
│   ├── results_comparison.png    # 性能对比图
│   └── results_boxplot.png       # 性能分布箱线图
│
└── outputs/                      # 模型输出目录
    ├── rt_layernorm_seed42/      # 各实验的模型和日志
    ├── rt_rmsnorm_seed42/
    ├── ...
```

## 自定义实验

如果您想进一步自定义实验参数，可以：

1. 修改`run_batch_experiments.bat`中的参数列表：
   ```batch
   set datasets=rt sst5
   set norm_types=layernorm rmsnorm
   set seeds=42 123 456 789 1024
   ```

2. 通过菜单界面的"自定义实验参数"选项

## 注意事项

1. 确保已安装所有必要的Python依赖，可以通过以下命令安装：
   ```
   pip install -r requirements.txt
   ```

2. 对于结果分析，需要额外安装pandas, matplotlib和openpyxl：
   ```
   pip install pandas matplotlib openpyxl
   ```

3. 运行完整的批量实验可能需要较长时间，建议在计算资源充足的情况下进行

4. 如果您只想测试特定组合，建议使用菜单界面的自定义实验选项 