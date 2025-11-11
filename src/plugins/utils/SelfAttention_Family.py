import torch
import torch.nn as nn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=1, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.output_attention = output_attention

    def forward(self, x):
        # 占位，真正注意力在 AttentionLayer 内部用 MHA 实现
        return x, None

class AttentionLayer(nn.Module):
    """
    简化版：用 nn.MultiheadAttention 实现自注意力（batch-first）
    输入/输出: [B, S, d_model]
    """
    def __init__(self, attention: FullAttention, d_model: int, n_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        y, _ = self.mha(x, x, x, need_weights=False)
        y = self.dropout(y)
        return self.norm(x + y), None
