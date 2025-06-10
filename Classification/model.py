import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0.2):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区，不会成为模型参数
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)
    https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        # self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        # 计算均方根
        ms = torch.mean(x ** 2, dim=-1, keepdim=True)
        # 标准化
        x_norm = x * torch.rsqrt(ms + self.eps)
        # 不再使用缩放
        return x_norm
        # return x_norm* self.scale

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dropout1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, dropout=0.2, norm_type='layernorm'):
        super(TransformerEncoderLayer, self).__init__()
        
        # 多头自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=dim_feedforward,
            out_features=d_model,
            act_layer=nn.ReLU,
            drop=dropout
        )
        
        # 归一化层 (LayerNorm 或 RMSNorm)
        if norm_type.lower() == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:  # 默认使用 LayerNorm
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-Norm架构
        # 第一个子层：多头自注意力
        src2 = self.norm1(src)
        src2 = src2.transpose(0, 1)  # 转换维度为 [seq_len, batch_size, d_model]
        src2, _ = self.self_attn(src2, src2, src2, 
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = src2.transpose(0, 1)  # 转换回 [batch_size, seq_len, d_model]
        src = src + self.dropout(src2)
        
        # 第二个子层：前馈网络
        src2 = self.norm2(src)
        src2 = self.mlp(src2)
        src = src + self.dropout(src2)
        
        return src

class SentimentClassifier(nn.Module):
    def __init__(self, 
            vocab_size, 
            num_classes, 
            d_model=128, 
            nhead=8, 
            dim_feedforward=512, 
            num_layers=4, 
            dropout=0.2, 
            max_len=128,
            norm_type='layernorm'):
        super(SentimentClassifier, self).__init__()
        
        # 保存使用的归一化类型
        self.norm_type = norm_type
        
        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_type)
            for _ in range(num_layers)
        ])
        
        # 添加分类器层
        self.classifier = nn.Linear(d_model, num_classes)
        
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        
        # 转换padding mask为key_padding_mask格式 (1 for padding, 0 for valid)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        # Embedding
        x = self.embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器层
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        
        # 取[CLS]位置的表示（假设是第一个token）
        x = x[:, 0, :]
        
        # 分类
        logits = self.classifier(x)
        
        return logits 