# inference_api.py
# 轻量推理接口：前端调用 -> 返回今日投资建议字符串（BUY/SELL/HOLD + position + confidence）

from typing import Union, List, Dict
import torch
import numpy as np
import pandas as pd
import logging

# === 依赖（与训练一致） ===
from .dualpath_model import DualPathTimesNetITrAligned
from .policy import GateActorCritic

# ---------- 与训练一致的超参 ----------
SEQ_LEN = 1440        # 观测窗口长度
RET_WINDOW = 60       # 仅用于构建骨干
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 预处理 ----------
def _clean_and_norm_window(df: pd.DataFrame) -> np.ndarray:
    """最近 SEQ_LEN 行做逐列 z-score，返回 [1, L, C]"""
    use_cols = ["open","high","low","close","volume"]
    x = df[use_cols].values.astype("float32")
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - m) / s
    return x[None, ...]   # [1, L, C]

def _build_backbone(seq_len: int = SEQ_LEN, ret_window: int = RET_WINDOW) -> torch.nn.Module:
    """与训练期一致的 TimesNet/dualpath 骨干（推理时只提特征）"""
    m = DualPathTimesNetITrAligned(
        seq_len=seq_len, pred_len=ret_window, enc_in=5, c_out=None,
        d_model=128, d_ff=256, e_layers=2,
        top_k=4, num_kernels=6, embed='timeF', freq='m',
        dropout=0.1, n_heads=4, factor=1, activation='gelu'
    ).to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m

def _extract_state(backbone: torch.nn.Module, x_ohlcv: np.ndarray) -> np.ndarray:
    """
    用骨干提取状态: concat(caps[-1], itr[-1]) -> [D]
    x_ohlcv: [1, L, C]  -> return [D,]
    """
    with torch.no_grad():
        xt = torch.from_numpy(x_ohlcv).to(DEVICE)           # [1, L, C]
        caps, itr = backbone(xt, x_mark=None)[:2]           # [1, T, H], [1, T, H]
        state = torch.cat([caps[:, -1, :], itr[:, -1, :]], dim=-1)  # [1, 2H]
        return state.squeeze(0).detach().cpu().numpy()       # [D]

# ---------- 策略推理 ----------
def _load_policy(state_dim: int, model_path: str) -> torch.nn.Module:
    """
    加载 GateActorCritic 的权重。
    兼容：
      - 直接 state_dict
      - {'state_dict': ...} 或 {'model': ...} 的包裹
    """
    net = GateActorCritic(in_dim=state_dim, hidden=128).to(DEVICE).eval()
    ckpt = torch.load(model_path, map_location=DEVICE)
    if isinstance(ckpt, dict):
        # 兼容常见保存格式
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt
    net.load_state_dict(sd, strict=False)  # 避免无关键报错
    return net

def _infer_position(policy: torch.nn.Module, state_vec: np.ndarray) -> float:
    """
    用 GateActorCritic 前向得到 gate 与连续动作。推理采用确定性策略：
      - gate_prob < 0.5 -> position = 0 (HOLD)
      - 否则 position = tanh(mu_raw)
    """
    with torch.no_grad():
        s = torch.from_numpy(state_vec).to(DEVICE).float().unsqueeze(0)  # [1, D]
        gate_logit, gate_prob, mu_raw, std, value = policy(s)
        gate_p = torch.sigmoid(gate_logit) if gate_prob is None else gate_prob
        if float(gate_p.squeeze(0)) < 0.5:
            return 0.0
        pos = torch.tanh(mu_raw).squeeze(0)  # 均值作动作
        return float(pos.item())

def _make_string_advice(pos: float) -> str:
    """把连续仓位转成文案"""
    if pos > 0.05:
        side = "BUY"
    elif pos < -0.05:
        side = "SELL"
    else:
        side = "HOLD"
    conf = min(0.99, abs(pos))
    return f"{side} | position={pos:.2f} | confidence={conf:.2f}"

# ================= 核心接口：前端直接调用 =================
logger = logging.getLogger("invest_plugin")

def get_invest_advice_from_ohlcv(
    ohlcv: Union[pd.DataFrame, List[Dict[str, float]]],
    model_path: str = "policy_final.pt",
    seq_len: int = SEQ_LEN
) -> str:
    """
    输入：最近 seq_len 分钟的 OHLCV
      - DataFrame: 需包含 ['ts','open','high','low','close','volume'] 列，时间升序
      - 或 list[dict]：含同名键
    输出：字符串，例如 "BUY | position=0.37 | confidence=0.37"
    """
    logger.info("开始分析投资建议")

    # 1) 准备数据
    df = pd.DataFrame(ohlcv) if not isinstance(ohlcv, pd.DataFrame) else ohlcv.copy()

    logger.info(f"输入类型: {type(ohlcv)}")
    logger.info(f"数据总行数: {len(df)}")
    logger.info(f"数据列名: {list(df.columns)}")
    logger.info(f"时间戳范围: {df['ts'].iloc[0]} → {df['ts'].iloc[-1]}")
    logger.info(f"价格范围 (close): {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    logger.info(f"成交量范围: {df['volume'].min():.2f} ~ {df['volume'].max():.2f}")

    need = ["ts", "open", "high", "low", "close", "volume"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")

    df = df.sort_values("ts").reset_index(drop=True)
    if len(df) < seq_len:
        raise ValueError(f"数据长度不足：需要≥{seq_len} 行，当前 {len(df)} 行。")

    window = df.iloc[-seq_len:].copy()

    logger.info(f"取最后 {seq_len} 条数据作为输入窗口")
    logger.info(f"窗口 close 范围: {window['close'].iloc[0]:.2f} → {window['close'].iloc[-1]:.2f}")
    logger.info(f"窗口 volume 均值: {window['volume'].mean():.2f}, std: {window['volume'].std():.2f}")

    x = _clean_and_norm_window(window)  # [1, L, C]

    logger.info(f"归一化后输入 shape: {x.shape}")
    logger.info(f"归一化后 close (z-score): [{x[0, :, 3].min():.3f}, {x[0, :, 3].max():.3f}]")
    logger.info(f"归一化后 volume (z-score): [{x[0, :, 4].min():.3f}, {x[0, :, 4].max():.3f}]")

    # 2) 提特征 -> 状态
    backbone = _build_backbone(seq_len=seq_len, ret_window=RET_WINDOW)
    state = _extract_state(backbone, x)  # [D]

    logger.info(f"提取的状态向量维度: {state.shape}")
    logger.info(f"状态向量均值: {state.mean().item():.3f}, std: {state.std().item():.3f}")
    logger.info(f"状态向量前5个值: {state[:5]}")

    # 3) 策略推理
    policy = _load_policy(state_dim=state.shape[0], model_path=model_path)
    pos = _infer_position(policy, state)

    logger.info(f"模型输出原始 position: {pos}")

    # 4) 输出文案
    advice = _make_string_advice(pos)
    logger.info(f"最终建议: {advice}")

    return advice
# def get_invest_advice_from_ohlcv(
#     ohlcv: Union[pd.DataFrame, List[Dict[str, float]]],
#     model_path: str = "policy_final.pt",
#     seq_len: int = SEQ_LEN
# ) -> str:
#     """
#     输入：最近 seq_len 分钟的 OHLCV
#       - DataFrame: 需包含 ['ts','open','high','low','close','volume'] 列，时间升序
#       - 或 list[dict]：含同名键
#     输出：字符串，例如 "BUY | position=0.37 | confidence=0.37"
#     """
#     # 1) 准备数据
#     df = pd.DataFrame(ohlcv) if not isinstance(ohlcv, pd.DataFrame) else ohlcv.copy()
#     need = ["ts","open","high","low","close","volume"]
#     for c in need:
#         if c not in df.columns:
#             raise ValueError(f"缺少列: {c}")
#     df = df.sort_values("ts").reset_index(drop=True)
#     if len(df) < seq_len:
#         raise ValueError(f"数据长度不足：需要≥{seq_len} 行，当前 {len(df)} 行。")

#     window = df.iloc[-seq_len:].copy()
#     x = _clean_and_norm_window(window)                       # [1, L, C]

#     # 2) 提特征 -> 状态
#     backbone = _build_backbone(seq_len=seq_len, ret_window=RET_WINDOW)
#     state = _extract_state(backbone, x)                      # [D]

#     # 3) 策略推理
#     policy = _load_policy(state_dim=state.shape[0], model_path=model_path)
#     pos = _infer_position(policy, state)

#     # 4) 输出文案
#     return _make_string_advice(pos)

# ============ 可选：返回结构化字典 ============
def get_invest_dict_from_ohlcv(
    ohlcv: Union[pd.DataFrame, List[Dict[str, float]]],
    model_path: str = "policy_final.pt",
    seq_len: int = SEQ_LEN
) -> Dict[str, float]:
    s = get_invest_advice_from_ohlcv(ohlcv, model_path, seq_len)
    side = s.split("|")[0].strip()
    pos = float(s.split("position=")[1].split("|")[0].strip())
    conf = float(s.split("confidence=")[1].strip())
    return {"advice": side, "position": pos, "confidence": conf}
