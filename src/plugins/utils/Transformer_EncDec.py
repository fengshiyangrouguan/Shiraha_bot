# layers/Transformer_EncDec.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.attn = attention_layer  # 已封装好的自注意力层（返回 [B,S,D]）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        x, _ = self.attn(x)       # 自注意力
        y = self.ffn(x)           # 前馈
        return self.norm(x + y), None

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, _ = layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x, None
