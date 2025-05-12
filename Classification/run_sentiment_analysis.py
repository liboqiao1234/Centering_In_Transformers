import os
import argparse
import subprocess
import logging
import sys
import time

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def run_command(command):
    """运行命令并实时输出结果"""
    logger.info(f"执行: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    return process.returncode

def check_requirements():
    """检查必要的依赖是否已安装"""
    required_packages = [
        "torch", 
        "transformers", 
        "scikit-learn", 
        "pandas", 
        "numpy", 
        "matplotlib", 
        "seaborn", 
        "tqdm",
        "wandb"  # 添加wandb依赖
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"缺少必要的依赖: {', '.join(missing_packages)}")
        logger.info("尝试安装缺失的依赖...")
        
        try:
            import pip
            for package in missing_packages:
                logger.info(f"安装 {package}...")
                run_command(f"{sys.executable} -m pip install {package}")
            logger.info("所有依赖安装完成")
        except Exception as e:
            logger.error(f"安装依赖失败: {e}")
            logger.error("请手动安装依赖: pip install " + " ".join(missing_packages))
            return False
    
    return True

def get_script_dir():
    """获取当前脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description="情感分析一键运行脚本")
    parser.add_argument("--dataset", type=str, default="rt", choices=["rt", "sst5"], help="数据集名称")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--skip_download", action="store_true", help="跳过数据下载步骤")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练步骤")
    parser.add_argument("--skip_evaluate", action="store_true", help="跳过评估步骤")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU进行训练")
    parser.add_argument("--install_deps", action="store_true", help="安装必要的依赖")
    parser.add_argument("--norm_type", type=str, default="layernorm", choices=["layernorm", "rmsnorm"], help="归一化类型")
    # Wandb参数
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="Transformer-Text-Classification", help="Wandb项目名称")
    parser.add_argument("--wandb_login", type=str, help="Wandb API密钥（可选）")
    
    args = parser.parse_args()
    
    # 检查/安装依赖
    if args.install_deps:
        check_requirements()
    
    # 如果启用wandb，检查是否已登录
    if args.use_wandb:
        # 尝试导入wandb
        try:
            import wandb
            logger.info("Wandb已正确安装")
            
            # 如果提供了API密钥，尝试登录
            if args.wandb_login:
                logger.info("正在使用提供的API密钥登录Wandb...")
                wandb_login_cmd = f"{sys.executable} -m wandb login {args.wandb_login}"
                if run_command(wandb_login_cmd) != 0:
                    logger.warning("Wandb登录失败，可能需要手动登录")
            
            # 尝试验证当前登录状态
            wandb_status_cmd = f"{sys.executable} -m wandb status"
            if run_command(wandb_status_cmd) != 0:
                logger.warning("Wandb未登录或状态检查失败")
                if not args.wandb_login:
                    logger.warning("请先登录Wandb或使用--wandb_login提供API密钥")
                    wandb_login_prompt = f"{sys.executable} -m wandb login"
                    run_command(wandb_login_prompt)
        except ImportError:
            logger.error("未安装Wandb，请先安装: pip install wandb")
            args.use_wandb = False
    
    # 确保使用绝对路径
    script_dir = get_script_dir()
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    # 创建目录
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 环境检查
    try:
        import torch
        has_gpu = torch.cuda.is_available() and not args.cpu
        logger.info(f"GPU可用性: {'是' if has_gpu else '否'}")
        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name}, 显存: {gpu_memory:.2f}GB")
    except:
        has_gpu = False
        logger.warning("无法检测GPU状态，将使用CPU")
    
    # 步骤1: 下载数据
    if not args.skip_download:
        logger.info(f"步骤1: 下载数据集 {args.dataset}...")
        
        # 构建命令，使用完整路径
        download_script = os.path.join(script_dir, "download_data.py")
        download_cmd = f"{sys.executable} '{download_script}' --dataset {args.dataset} --data_dir '{args.data_dir}'"
        
        # 设置重试次数
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            if retry_count > 0:
                logger.info(f"重试下载 (尝试 {retry_count+1}/{max_retries})...")
                time.sleep(2 ** retry_count)  # 指数退避
            
            result = run_command(download_cmd)
            if result == 0:
                success = True
            else:
                logger.warning(f"数据下载命令返回错误代码: {result}")
                retry_count += 1
        
        if not success:
            logger.error("数据下载失败，退出程序")
            return
    else:
        logger.info("跳过数据下载步骤")
    
    # 步骤2: 训练模型
    if not args.skip_train:
        logger.info(f"步骤2: 训练 {args.dataset} 模型...")
        
        # 构建命令，使用完整路径
        train_script = os.path.join(script_dir, "train.py")
        train_cmd = f"{sys.executable} '{train_script}' --dataset {args.dataset} --data_dir '{args.data_dir}' --output_dir '{args.output_dir}' --epochs {args.epochs}"
        
        # 添加wandb参数
        if args.use_wandb:
            train_cmd += f" --use_wandb --wandb_project '{args.wandb_project}'"
        train_cmd += f" --norm_type {args.norm_type}"

        if args.cpu:
            # 设置环境变量强制使用CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("强制使用CPU进行训练")
        
        if run_command(train_cmd) != 0:
            logger.error("模型训练失败，退出程序")
            return
    else:
        logger.info("跳过模型训练步骤")
    
    # 步骤3: 评估模型
    if not args.skip_evaluate:
        model_path = os.path.join(args.output_dir, f"{args.dataset}_best.pt")
        if os.path.exists(model_path):
            logger.info(f"步骤3: 评估 {args.dataset} 模型...")
            
            # 构建命令，使用完整路径
            eval_script = os.path.join(script_dir, "evaluate.py")
            eval_cmd = f"{sys.executable} '{eval_script}' --dataset {args.dataset} --data_dir '{args.data_dir}' --model_path '{model_path}' --output_dir '{args.output_dir}'"
            
            # 添加wandb参数
            if args.use_wandb:
                eval_cmd += f" --use_wandb --wandb_project '{args.wandb_project}'"
            
            if args.cpu:
                # 设置环境变量强制使用CPU
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                logger.info("强制使用CPU进行评估")
            eval_cmd += f" --norm_type {args.norm_type}"
            if run_command(eval_cmd) != 0:
                logger.error("模型评估失败")
            else:
                logger.info(f"情感分析模型全流程完成！可以使用以下命令进行交互式预测：")
                
                inference_script = os.path.join(script_dir, "inference.py")
                inference_cmd = f"{sys.executable} '{inference_script}' --model_path '{model_path}' --dataset {args.dataset}"
                
                logger.info(f"{inference_cmd}")
        else:
            logger.error(f"找不到模型文件: {model_path}，跳过评估步骤")
    else:
        logger.info("跳过模型评估步骤")

if __name__ == "__main__":
    main() 