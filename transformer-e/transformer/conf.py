"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import torch

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

# 训练参数
batch_size = 128
max_len = 100
epoch = 100
warmup = 20
init_lr = 1e-4
factor = 0.9
adam_eps = 5e-9
patience = 5
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
