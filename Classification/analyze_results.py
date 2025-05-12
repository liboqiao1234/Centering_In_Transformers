import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_result_file(file_path):
    """解析单个结果文件，提取准确率和F1分数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 提取测试集结果
            accuracy_match = re.search(r'测试集.*?Accuracy: ([\d\.]+)', content)
            f1_match = re.search(r'测试集.*?Macro-F1: ([\d\.]+)', content)
            
            if accuracy_match and f1_match:
                accuracy = float(accuracy_match.group(1))
                f1 = float(f1_match.group(1))
                return accuracy, f1
            else:
                return None, None
    except Exception as e:
        print(f"无法解析文件 {file_path}: {e}")
        return None, None

def analyze_results(results_dir):
    """分析实验结果目录中的所有结果文件"""
    results = []
    
    # 遍历结果文件
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.txt'):
            # 从文件名解析参数
            parts = filename.replace('_results.txt', '').split('_')
            if len(parts) >= 3:
                dataset = parts[0]
                norm_type = parts[1]
                seed = parts[2].replace('seed', '')
                
                file_path = os.path.join(results_dir, filename)
                accuracy, f1 = parse_result_file(file_path)
                
                if accuracy is not None and f1 is not None:
                    results.append({
                        '数据集': dataset,
                        '归一化类型': norm_type,
                        '随机种子': seed,
                        '准确率': accuracy,
                        'F1分数': f1
                    })
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='分析批量实验结果')
    parser.add_argument('--results_dir', type=str, default='experiment_results', help='实验结果目录')
    parser.add_argument('--output_file', type=str, default='experiment_results/analysis_results.xlsx', help='结果输出文件')
    args = parser.parse_args()
    
    # 确保结果目录存在
    if not os.path.exists(args.results_dir):
        print(f"错误: 结果目录 {args.results_dir} 不存在")
        return
    
    # 分析结果
    df = analyze_results(args.results_dir)
    
    if df.empty:
        print("未找到有效的结果文件")
        return
    
    # 保存原始数据到Excel
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_excel(args.output_file, index=False)
    print(f"原始数据已保存到 {args.output_file}")
    
    # 按数据集和归一化类型分组统计
    grouped_stats = df.groupby(['数据集', '归一化类型']).agg({
        '准确率': ['mean', 'std', 'min', 'max'],
        'F1分数': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("\n=== 实验结果统计 ===")
    print(grouped_stats)
    
    # 将统计结果保存到Excel的另一个表
    with pd.ExcelWriter(args.output_file, engine='openpyxl', mode='a') as writer:
        grouped_stats.to_excel(writer, sheet_name='统计结果')
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 准确率对比图
    plt.subplot(2, 1, 1)
    for (dataset, norm_type), group in df.groupby(['数据集', '归一化类型']):
        label = f"{dataset}-{norm_type}"
        plt.plot(group['随机种子'], group['准确率'], 'o-', label=label)
    
    plt.title('不同数据集和归一化类型的准确率对比')
    plt.xlabel('随机种子')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    # F1分数对比图
    plt.subplot(2, 1, 2)
    for (dataset, norm_type), group in df.groupby(['数据集', '归一化类型']):
        label = f"{dataset}-{norm_type}"
        plt.plot(group['随机种子'], group['F1分数'], 'o-', label=label)
    
    plt.title('不同数据集和归一化类型的F1分数对比')
    plt.xlabel('随机种子')
    plt.ylabel('F1分数')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'results_comparison.png'))
    print(f"对比图已保存到 {os.path.join(args.results_dir, 'results_comparison.png')}")
    
    # 创建箱线图比较不同归一化方法
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    boxplot_data = [df[(df['数据集'] == dataset) & (df['归一化类型'] == norm_type)]['准确率'] 
                  for dataset in df['数据集'].unique() 
                  for norm_type in df['归一化类型'].unique()]
    boxplot_labels = [f"{dataset}-{norm_type}" 
                    for dataset in df['数据集'].unique() 
                    for norm_type in df['归一化类型'].unique()]
    plt.boxplot(boxplot_data, labels=boxplot_labels)
    plt.title('不同实验设置的准确率分布')
    plt.ylabel('准确率')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    boxplot_data = [df[(df['数据集'] == dataset) & (df['归一化类型'] == norm_type)]['F1分数'] 
                  for dataset in df['数据集'].unique() 
                  for norm_type in df['归一化类型'].unique()]
    boxplot_labels = [f"{dataset}-{norm_type}" 
                    for dataset in df['数据集'].unique() 
                    for norm_type in df['归一化类型'].unique()]
    plt.boxplot(boxplot_data, labels=boxplot_labels)
    plt.title('不同实验设置的F1分数分布')
    plt.ylabel('F1分数')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'results_boxplot.png'))
    print(f"箱线图已保存到 {os.path.join(args.results_dir, 'results_boxplot.png')}")

if __name__ == '__main__':
    main() 