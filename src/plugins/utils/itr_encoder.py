import torch
import torch.nn as nn
from .Embed import DataEmbedding_inverted
from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer

class ITrEncoderOnly(nn.Module):
    """iTransformer encoder-only：输入 [B,L_out,C] → 输出 [B,L_out,d_model]"""
    def __init__(self, seq_len, d_model, n_heads, d_ff, e_layers, embed='timeF', freq='h', dropout=0.1, factor=1, activation='gelu'):
        super().__init__()
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)
        self.encoder = Encoder(
            [EncoderLayer(AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                         d_model, n_heads),
                          d_model, d_ff, dropout=dropout, activation=activation)
             for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )

    def forward(self, x, x_mark=None):
        # enc_in: [B, C_in, d_model]（把通道当作“序列长度”）
        enc_in = self.enc_embedding(x, x_mark)
        enc_out, _ = self.encoder(enc_in, attn_mask=None)  # [B, C_in, d_model]
        # 与上游保持一致：返回 [B, L_out, d_model]
        # 这里 L_out = x.size(1)；我们简单把每个时间步的特征设为通道聚合后的同一表征
        # 为了形状对齐，扩展/广播到 L_out
        B, C_in, D = enc_out.shape
        L_out = x.size(1)
        # 重复到 L_out
        return enc_out.mean(dim=1, keepdim=True).repeat(1, L_out, 1)
