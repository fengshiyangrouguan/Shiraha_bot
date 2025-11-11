# models/dualpath_model.py
import torch
import torch.nn as nn
from .Embed import DataEmbedding
from .times_blocks import TimesBlockExpose
from .capsules import PeriodCapsuleAggregator
from .itr_encoder import ITrEncoderOnly

class DualPathTimesNetITrAligned(nn.Module):
    """
    返回：
      feat_caps [B, L_out, C_caps] （短期专家）
      feat_itr  [B, L_out, d_model]（长期/风险专家）
      fuse      [B, L_out, C_caps+d_model]
      pred      [B, L_out, c_out]（若启用监督预测头）
    """
    def __init__(self, seq_len, pred_len, enc_in, c_out,
                 d_model=256, d_ff=512, e_layers=2, top_k=3, num_kernels=6,
                 embed="timeF", freq="h", dropout=0.1, n_heads=8, factor=1, activation="gelu"):
        super().__init__()
        self.L, self.P = seq_len, pred_len
        self.task_is_pred = c_out is not None

        self.enc = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # TimesBlock 的 pred_len 仅在对齐到 L+P（align 启用）时才生效
        self.align = None
        if self.task_is_pred and self.P > 0:
            self.align = nn.Linear(self.L, self.L + self.P)

        tb_pred_len = self.P if (self.align is not None) else 0
        tb_cfg = type("CFG",(object,),dict(
            seq_len=self.L, pred_len=tb_pred_len, top_k=top_k,
            d_model=d_model, d_ff=d_ff, num_kernels=num_kernels
        ))
        self.blocks = nn.ModuleList([TimesBlockExpose(tb_cfg) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)

        L_out = (self.L + self.P) if (self.align is not None) else self.L
        self.caps = PeriodCapsuleAggregator(d_model, P=6, Dp=8, D=4, Dd=12, iters=2)
        self.itr  = ITrEncoderOnly(seq_len=L_out, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                   e_layers=e_layers, embed=embed, freq=freq,
                                   dropout=dropout, factor=factor, activation=activation)

        self.caps_dim = 4*12
        self.d_model  = d_model
        self.proj = nn.Linear(d_model, c_out) if self.task_is_pred else None

    def _norm(self, x):
        m = x.mean(1, keepdim=True).detach()
        s = (x.var(1, keepdim=True, unbiased=False)+1e-5).sqrt()
        return (x-m)/s, m, s

    def _denorm(self, y, m, s, L_out):
        return y * s[:,0,:].unsqueeze(1).repeat(1,L_out,1) + m[:,0,:].unsqueeze(1).repeat(1,L_out,1)

    def forward(self, x, x_mark=None):
        # x: [B,L,C]
        x, m, s = self._norm(x)
        h = self.enc(x, x_mark)                    # [B,L,d_model]
        if self.align is not None:
            h = self.align(h.permute(0,2,1)).permute(0,2,1)  # [B,L+P,d_model]

        res_soft = h
        for blk in self.blocks:
            _, res_soft = blk(res_soft)            # 级联 TimesBlock，取 soft 融合
            res_soft = self.norm(res_soft)

        L_out = res_soft.size(1)
        # 为胶囊需要的 res_stack：重算一次（只需最后一层的 stack）
        res_stack, _ = self.blocks[-1](res_soft)   # [B,L_out,d_model,k]
        feat_caps = self.caps(res_stack)           # [B,L_out,C_caps]
        feat_itr  = self.itr(res_soft, x_mark)     # [B,L_out,d_model]
        fuse = torch.cat([feat_caps, feat_itr], dim=-1)

        pred = self._denorm(self.proj(feat_itr), m, s, L_out) if self.proj is not None else None
        return feat_caps, feat_itr, fuse, pred
