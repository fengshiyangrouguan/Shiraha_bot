# service.py
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import pandas as pd
import io

# 你自己的推理函数
from inference_api import (
    get_invest_advice_from_ohlcv,
    get_invest_dict_from_ohlcv,
    SEQ_LEN as DEFAULT_SEQ_LEN,
)

# ============ 配置 ============
MODEL_PATH = Path(__file__).parent / "policy_final.pt"
DEFAULT_SEQ = int(os.environ.get("SEQ_LEN", DEFAULT_SEQ_LEN))

app = FastAPI(title="RL Invest Advice API",
              version="1.0",
              description="Return today's investment advice from OHLCV window")

# ============ 入参模型（JSON） ============
class OHLCVPoint(BaseModel):
    ts: Any
    open: float
    high: float
    low: float
    close: float
    volume: float

class JSONPayload(BaseModel):
    data: List[OHLCVPoint]
    seq_len: Optional[int] = None
    # 如果需要不同模型文件，可提供；否则用默认
    model_path: Optional[str] = None
    # 返回模式："string" 或 "dict"
    return_mode: Optional[str] = "string"

# ============ 健康检查 ============
@app.get("/health")
def health():
    return {"status": "ok"}

# ============ JSON 推理（推荐给前端） ============
@app.post("/predict")
def predict_json(payload: JSONPayload):
    try:
        df = pd.DataFrame([p.dict() for p in payload.data])
        # 只取前6列（容错：多给列也无妨）
        need = ["ts", "open", "high", "low", "close", "volume"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"缺少列: {missing}")

        df = df[need].sort_values("ts").reset_index(drop=True)

        seq_len = payload.seq_len or DEFAULT_SEQ
        model_path = payload.model_path or MODEL_PATH

        if payload.return_mode == "dict":
            res = get_invest_dict_from_ohlcv(df, model_path=model_path, seq_len=seq_len)
            return JSONResponse(res)
        else:
            s = get_invest_advice_from_ohlcv(df, model_path=model_path, seq_len=seq_len)
            return PlainTextResponse(s)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict_json error: {e}")

# ============ CSV 推理（上传文件） ============
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...),
                      seq_len: Optional[int] = Body(None),
                      model_path: Optional[str] = Body(None),
                      return_mode: Optional[str] = Body("string")):
    """
    - file: CSV 文件，前6列为 ts, open, high, low, close, volume
    - seq_len: 可覆盖默认窗口长度
    - return_mode: "string" 或 "dict"
    - model_path: 可覆盖默认权重路径
    """
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), usecols=range(6), header=0)
        df.columns = ["ts", "open", "high", "low", "close", "volume"]
        df = df.sort_values("ts").reset_index(drop=True)

        seq = seq_len or DEFAULT_SEQ
        mpath = model_path or MODEL_PATH

        if return_mode == "dict":
            res = get_invest_dict_from_ohlcv(df, model_path=mpath, seq_len=seq)
            return JSONResponse(res)
        else:
            s = get_invest_advice_from_ohlcv(df, model_path=mpath, seq_len=seq)
            return PlainTextResponse(s)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict_csv error: {e}")
