import torch
import torch.nn as nn

class DataEmbedding(nn.Module):
    """
    把 [B, L, C] 投到 [B, L, d_model]
    """
    def __init__(self, c_in, d_model, embed="timeF", freq="h", dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        return self.dropout(self.proj(x))

class DataEmbedding_inverted(nn.Module):
    """
    把 [B, L, C] 映射为 [B, C, d_model]（把通道当序列）
    简化实现：对每个通道的 L 步用同一线性层映射到 d_model，然后拼回 [B, C, d_model]
    """
    def __init__(self, seq_len, d_model, embed="timeF", freq="h", dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):  # x: [B, L, C]
        B, L, C = x.shape
        xt = x.transpose(1, 2)          # [B, C, L]
        y = self.fc(xt)                 # [B, C, d_model]
        return self.dropout(y)
