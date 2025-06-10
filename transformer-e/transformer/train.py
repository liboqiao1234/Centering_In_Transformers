"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import wandb
import math
import time
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from torch import optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from conf import *
from Taiyi.taiyi.monitor import Monitor
from Taiyi.visualize import Visualization


step = 0
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        # 使用Xavier均匀初始化代替Kaiming初始化
        nn.init.xavier_uniform_(m.weight.data)
        
    # 初始化Transformer中的偏置
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, 0)


# 预热+线性递减学习率调度器
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """
    实现预热+线性递减的学习率调度策略
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 预热期间线性增加
            return float(current_step) / float(max(1, warmup_steps))
        # 预热后线性递减
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(model, iterator, optimizer, criterion, scheduler, monitor=None, vis_wandb=None):
    global step
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        if monitor is not None:
            monitor.track(step)
        if vis_wandb is not None:
            vis_wandb.show(step)
        
        loss.backward()
        
        # 添加梯度裁剪，参考旧版代码
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        # 每个batch更新学习率
        scheduler.step()
        
        epoch_loss += loss.item()

        # 记录每个batch的loss到wandb
        if (i+1) % 20 == 0:  # 每20个batch记录一次，减少IO负担
            wandb.log({
                "batch_loss": loss.item(), 
                "step": step,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            print(f'step: {i+1}/{len(iterator)}, loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]["lr"]:.8f}')
            
        step += 1

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass
            if len(total_bleu) > 0:
                batch_bleu.append(sum(total_bleu) / len(total_bleu))

    return epoch_loss / len(iterator), sum(batch_bleu) / len(batch_bleu) if len(batch_bleu) > 0 else 0


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Transformer Translation Model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--norm-type', type=str, default='ln', choices=['ln', 'rms'], 
                       help='Normalization type: ln for LayerNorm, rms for RMSNorm')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ffn-hidden', type=int, default=2048, help='FFN hidden dimension')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='预热步数占总步数的比例')
    args = parser.parse_args()
    
    # 设置随机种子，确保实验可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 初始化 wandb
    global step
    step = 0
    
    # 处理norm_type参数
    if args.norm_type == 'ln':
        norm_type = 'LayerNorm'  # 原来的代码需要完整名称
    else:
        norm_type = 'RMS'  # 原来的代码是使用RMS
    
    # 设置全局梯度裁剪值
    global clip
    clip = args.clip
    
    print("\n===== 训练参数 =====")
    print(f"随机种子: {args.seed}")
    print(f"归一化层类型: {args.norm_type}")
    print(f"初始学习率: {args.learning_rate}")
    print(f"训练轮数: {args.epochs}")
    print(f"预热比例: {args.warmup_ratio}")
    print(f"批量大小: {args.batch_size}")
    print(f"梯度裁剪值: {args.clip}")
    print(f"模型维度: {args.d_model}")
    print(f"前馈网络维度: {args.ffn_hidden}")
    print(f"注意力头数: {args.n_heads}")
    print(f"编码器/解码器层数: {args.n_layers}")
    print(f"Dropout概率: {args.dropout}")
    print("=====================\n")
    
    wandb.init(
        project="transformer-translation",
        name=f"transformer-translation_e{args.epochs}_b{args.batch_size}_d{args.d_model}_n{args.n_layers}_h{args.n_heads}_f{args.ffn_hidden}_{args.norm_type}_seed{args.seed}_lr{args.learning_rate}",
        config={
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "drop_prob": args.dropout,
            "epoch": args.epochs,
            "init_lr": args.learning_rate,
            "norm": args.norm_type,
            "seed": args.seed,
            "warmup_ratio": args.warmup_ratio,
            "clip": args.clip
        }
    )

    taiyi_config = {
        "PositionwiseFeedForward": [['InputAngleStd','linear(5,0)'], ['InputAngleMean', 'linear(5,0)'], ['OutputAngleStd', 'linear(5,0)'], ['OutputAngleMean', 'linear(5,0)']],
        "self_attention": [['InputAngleStd','linear(5,0)'], ['InputAngleMean', 'linear(5,0)'], ['OutputAngleStd', 'linear(5,0)'], ['OutputAngleMean', 'linear(5,0)']],
        "LayerNorm": [['InputAngleStd','linear(5,0)'], ['InputAngleMean', 'linear(5,0)'], ['OutputAngleStd', 'linear(5,0)'], ['OutputAngleMean', 'linear(5,0)']],
        "RMSNorm": [['InputAngleStd','linear(5,0)'], ['InputAngleMean', 'linear(5,0)'], ['OutputAngleStd', 'linear(5,0)'], ['OutputAngleMean', 'linear(5,0)']],
    }
    
    # 初始化 tokenizer
    tokenizer = Tokenizer()

    # 加载数据
    loader = DataLoader(ext=('.en', '.de'),
                       tokenize_en=tokenizer.tokenize_en,
                       tokenize_de=tokenizer.tokenize_de,
                       init_token='<sos>',
                       eos_token='<eos>')

    train_data, valid_data, test_data = loader.make_dataset()
    loader.build_vocab(train_data, min_freq=2)
    train_iter, valid_iter, test_iter = loader.make_iter(train_data, valid_data, test_data,
                                                        batch_size=args.batch_size,
                                                        device=device)

    # 初始化模型
    model = Transformer(src_pad_idx=loader.source.vocab.stoi['<pad>'],
                       trg_pad_idx=loader.target.vocab.stoi['<pad>'],
                       trg_sos_idx=loader.target.vocab.stoi['<sos>'],
                       d_model=args.d_model,
                       enc_voc_size=enc_voc_size,
                       dec_voc_size=dec_voc_size,
                       max_len=max_len,
                       ffn_hidden=args.ffn_hidden,
                       n_head=args.n_heads,
                       n_layers=args.n_layers,
                       drop_prob=args.dropout,
                       device=device,
                       norm_type=norm_type).to(device)
    monitor = Monitor(model, taiyi_config)
    vis_wandb = Visualization(monitor, wandb)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.apply(initialize_weights)

    # 初始化优化器和损失函数
    optimizer = Adam(params=model.parameters(),
                     lr=args.learning_rate,
                     betas=(0.9, 0.98),  # 修改为Transformer论文中推荐的值
                     weight_decay=weight_decay,
                     eps=adam_eps)
    
    # 计算总步数和预热步数
    steps_per_epoch = len(train_iter)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 使用预热+线性递减的学习率调度器
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    print(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")

    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

    # 训练循环
    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, scheduler, monitor, vis_wandb)
        valid_loss, valid_bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 记录到 wandb
        wandb.log({
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_bleu": valid_bleu,
            "epoch": epoch + 1,
            "epoch_time": end_time - start_time,
            "learning_rate": current_lr
        })

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'saved/model-{args.norm_type}-seed{args.seed}-best.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_time(start_time, end_time)}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} | Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tVal BLEU: {valid_bleu:.3f}')
        print(f'\tLearning Rate: {current_lr:.8f}')

    # 测试模型
    test_loss, test_bleu = evaluate(model, test_iter, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU: {test_bleu:.3f} |')
    
    # 记录最终测试结果
    wandb.log({
        "test_loss": test_loss,
        "test_bleu": test_bleu
    })

    # 完成 wandb 运行
    wandb.finish()


if __name__ == '__main__':
    main()
