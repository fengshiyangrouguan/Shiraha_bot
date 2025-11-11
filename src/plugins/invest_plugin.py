# E:\project\Shiraha_bot\shirahabot\plugins\invest_plugin.py
import json
import os
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Type

from src.plugin_system.base import BaseTool, BasePlugin
from src.plugins.utils.inference_api import get_invest_dict_from_ohlcv

logger = logging.getLogger("main")

DEFAULT_SEQ = 60
MODEL_NAME = "policy_final.pt"
REQUIRED_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


class InvestAnalysisTool(BaseTool):
    """
    投资分析工具 - 基于本地 K 线 JSON 文件执行模型预测并生成投资建议
    """
    name = "invest_analysis"
    description = "根据本地股票历史数据生成投资分析建议。"
    available_for_llm = True

    parameters = []  # 不需要任何输入参数

    async def execute(self) -> Dict[str, Any]:
        logger.info("InvestAnalysisTool: 开始投资分析流程...")

        try:
            all_files = self._load_all_json()
        except Exception as e:
            return {"error": f"数据加载失败: {e}"}

        if not all_files:
            return {"error": "未找到可用数据，请检查 data/invest_data 目录"}

        # 只取第一个文件
        filename, rows = next(iter(all_files.items()))
        df = pd.DataFrame(rows)

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            return {"error": f"{filename} 缺少必要列: {missing_cols}"}

        df = df[REQUIRED_COLUMNS].sort_values("ts").reset_index(drop=True)
        seq_len =  DEFAULT_SEQ

        current_dir = Path(__file__).parent
        model_path = current_dir / "utils" / MODEL_NAME

        if len(df) < seq_len:
            return {"error": f"{filename} 数据长度不足（{len(df)} < {seq_len}）"}

        if not os.path.exists(model_path):
            return {"error": f"模型文件不存在: {model_path}"}

        # 调用模型
        advice_dict = get_invest_dict_from_ohlcv(df, model_path=model_path, seq_len=seq_len)

        # 将字典解析为自然语言 prompt
        # 假设 advice_dict = {"advice": "buy", "position": "long", "confidence": 0.85}
        advice_str = (
            f"投资分析结果:\n"
            f"- 建议: {advice_dict.get('advice', '未知')}\n"
            f"- 仓位: {advice_dict.get('position', '未知')}\n"
            f"- 置信度: {advice_dict.get('confidence', '未知')}"
        )

        return {"result": advice_str}

    def _load_all_json(self) -> Dict[str, Any]:
        """
        扫描 data/invest_data 目录下所有 .json 文件
        """
        data_dir = Path(__file__).parent.parent.parent / "data" / "invest_data"
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        result = {}
        for file in data_dir.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                result[file.name] = json.load(f)

        return result


class InvestPlugin(BasePlugin):
    def get_tools(self) -> List[Type[BaseTool]]:
        return [InvestAnalysisTool]
