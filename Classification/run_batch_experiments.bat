@echo off
setlocal enabledelayedexpansion

echo 开始运行批量实验...
echo ===========================================

REM 设置实验参数
set datasets=rt sst5
set norm_types=layernorm rmsnorm
set seeds=42 123 456 789 1024

REM 创建实验结果目录
if not exist "experiment_results" mkdir experiment_results

REM 记录开始时间
echo 批量实验开始时间: %date% %time% > experiment_results\experiment_log.txt

REM 遍历所有组合
for %%d in (%datasets%) do (
    for %%n in (%norm_types%) do (
        for %%s in (%seeds%) do (
            echo ===========================================
            echo 运行实验: 数据集=%%d, 归一化类型=%%n, 随机种子=%%s
            echo ===========================================
            
            REM 构建输出目录
            set output_dir=outputs\%%d_%%n_seed%%s
            
            REM 记录命令到日志
            echo 运行命令: python train.py --dataset %%d --norm_type %%n --seed %%s --output_dir !output_dir! >> experiment_results\experiment_log.txt
            
            REM 执行训练
            python train.py --dataset %%d --norm_type %%n --seed %%s --output_dir !output_dir!
            
            REM 记录完成状态
            if !errorlevel! equ 0 (
                echo [成功] 数据集=%%d, 归一化类型=%%n, 随机种子=%%s >> experiment_results\experiment_log.txt
            ) else (
                echo [失败] 数据集=%%d, 归一化类型=%%n, 随机种子=%%s >> experiment_results\experiment_log.txt
            )
            
            REM 保存评估结果到单独文件
            if exist "!output_dir!\eval_results.txt" (
                copy "!output_dir!\eval_results.txt" "experiment_results\%%d_%%n_seed%%s_results.txt" > nul
            )
        )
    )
)

REM 记录结束时间
echo 批量实验结束时间: %date% %time% >> experiment_results\experiment_log.txt

echo ===========================================
echo 所有实验已完成！结果保存在 experiment_results 目录中。
echo 请查看 experiment_results\experiment_log.txt 获取完整日志。
echo ===========================================

endlocal 