# models/times_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Conv_Blocks import Inception_Block_V1

@torch.no_grad()
def FFT_for_Period(x: torch.Tensor, k: int):
    """
    x: [B, T, C]
    返回:
      period_list: list[int]（有效 top-k 的周期，已截断在 [1, T]）
      period_weight: [B, k_eff]（融合权重的打分，未 softmax）
    """
    B, T, C = x.shape
    xf = torch.fft.rfft(x, dim=1)               # [B, F, C], F = T//2 + 1
    amp = xf.abs()
    score = amp.mean(0).mean(-1)                # [F]
    if score.numel() > 0:
        score = score.clone()
        score[0] = 0                            # 去直流
    Fbins = score.numel()
    k_eff = max(1, min(k, Fbins))
    _, top = torch.topk(score, k_eff)
    top = top.detach().cpu().tolist()           # 频点索引
    # 周期：防止 0 & 过大
    period_list = [max(1, min(T, T // (t + 1))) for t in top]
    period_weight = amp.mean(-1)[:, top]        # [B, k_eff]
    return period_list, period_weight

class TimesBlockExpose(nn.Module):
    """
    输出:
      res_stack: [B, L_out, C, k_eff]
      res_soft : [B, L_out, C]
    说明：
      - L_out = T = x.size(1)，始终按实际输入长度工作（不受 pred_len 影响）
    """
    def __init__(self, configs):
        super().__init__()
        d_model     = configs.d_model
        d_ff        = configs.d_ff
        num_kernels = getattr(configs, "num_kernels", 6)
        self.k      = getattr(configs, "top_k", 3)

        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        B, T, C = x.shape
        L_target = T

        period_list, period_weight = FFT_for_Period(x, self.k)
        k_eff = len(period_list)
        outs = []

        for p in period_list:
            p = int(max(1, min(T, int(p))))
            # 对齐到 L_target
            xx = x
            if xx.size(1) < L_target:
                pad0 = torch.zeros(B, L_target - xx.size(1), C, device=x.device, dtype=x.dtype)
                xx = torch.cat([xx, pad0], dim=1)
            # 再对齐到 p 的整数倍
            length = ((L_target + p - 1) // p) * p
            if xx.size(1) < length:
                pad1 = torch.zeros(B, length - xx.size(1), C, device=x.device, dtype=x.dtype)
                xx = torch.cat([xx, pad1], dim=1)

            # [B, length, C] -> [B, C, length//p, p]
            xx = xx.reshape(B, length // p, p, C).permute(0, 3, 1, 2).contiguous()
            xx = self.conv(xx)
            xx = xx.permute(0, 2, 3, 1).reshape(B, -1, C)[:, :L_target, :]
            outs.append(xx)

        res_stack = torch.stack(outs, dim=-1) if k_eff > 1 else outs[0].unsqueeze(-1)
        # 融合权重
        w = F.softmax(period_weight, dim=1) if k_eff > 0 else torch.ones(B,1,device=x.device,dtype=x.dtype)
        w = w[:, None, None, :].expand_as(res_stack)
        res_soft = (res_stack * w).sum(-1) + x
        res_stack = res_stack + x[:, :, :, None]
        return res_stack, res_soft

