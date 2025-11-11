# layers/Conv_Blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _resize_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # 统一空间尺寸，杜绝 off-by-one 导致的 cat 报错
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

class Inception_Block_V1(nn.Module):
    """
    多分支 1xK 卷积（在 W 维），再 1x1 投影。
    保险策略：
      1) Conv2d(..., padding='same') 尽量保持空间尺寸一致
      2) 若仍有 1px 偏差，插值到 (H_min, W_min) 再 cat
    输入:  [B, C, H, W]
    输出:  [B, out_channels, H, W]
    """
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()
        # 如要消除 padding='same' 的 warning，可改为全奇数核：[1,3,5,7,9,11]
        ks = [1, 2, 3, 5, 7, 11][:max(1, num_kernels)]
        per = max(1, out_channels // len(ks))
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, per, kernel_size=(1, k), stride=1, padding='same'),
                nn.GELU()
            ) for k in ks
        ])
        self.proj = nn.Conv2d(per * len(ks), out_channels, kernel_size=1)

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        H_min = min(o.shape[2] for o in outs)
        W_min = min(o.shape[3] for o in outs)
        outs = [_resize_to(o, H_min, W_min) for o in outs]
        o = torch.cat(outs, dim=1)
        return self.proj(o)


