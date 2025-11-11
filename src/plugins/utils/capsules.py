# models/capsules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

CAPSULES_IMPL_ID = "capsules_einsum_kdqp_to_blkdq_v1"

def squash(v, dim=-1, eps=1e-9):
    s2 = (v * v).sum(dim=dim, keepdim=True)
    return (s2 / (1.0 + s2)) * v / (s2.sqrt() + eps)

class PeriodCapsuleAggregator(nn.Module):
    """
    输入 X: [B, L, C, K]（来自 TimesBlock 的 res_stack）
    输出  : [B, L, D*Dd]
    - W: [K, D, Dd, P*Dp]，当 K 变化时重建
    - u_hat: einsum('kdqp, blkp -> blkdq')
    """
    def __init__(self, in_dim, P=6, Dp=8, D=4, Dd=12, iters=2):
        super().__init__()
        self.P, self.Dp, self.D, self.Dd, self.iters = P, Dp, D, Dd, iters
        self.primary = nn.Linear(in_dim, P * Dp)
        self.register_parameter("W", None)

    def _ensure_W(self, K, device, dtype):
        if (self.W is None) or (self.W.shape[0] != K):
            W = torch.empty(K, self.D, self.Dd, self.P * self.Dp, device=device, dtype=dtype)
            nn.init.xavier_uniform_(W)
            self.W = nn.Parameter(W)

    def forward(self, X):  # X: [B,L,C,K]
        B, L, C, K = X.shape
        self._ensure_W(K, X.device, X.dtype)

        # primary capsules: [B,L,K,P,Dp]
        Xp = self.primary(X.permute(0,1,3,2).contiguous().view(B, L*K, C)).view(B, L, K, self.P, self.Dp)
        Xp = squash(Xp, dim=-1)
        Xp_flat = Xp.view(B, L, K, self.P * self.Dp)  # [B,L,K,p]

        # u_hat: [B,L,K,D,Dd]，einsum: 'kdqp,blkp->blkdq'
        u_hat = torch.einsum('kdqp,blkp->blkdq', self.W, Xp_flat)

        # 动态路由
        b = torch.zeros(B, L, K, self.D, device=X.device, dtype=X.dtype)
        for _ in range(self.iters):
            c = F.softmax(b, dim=-1)                 # [B,L,K,D]
            s = (c.unsqueeze(-1) * u_hat).sum(2)     # [B,L,D,Dd]
            v = squash(s, dim=-1)                    # [B,L,D,Dd]
            b = b + (u_hat * v.unsqueeze(2)).sum(-1) # [B,L,K,D]

        return v.view(B, L, self.D * self.Dd)        # [B,L,D*Dd]
