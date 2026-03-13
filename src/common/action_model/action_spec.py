# src/common/action_model/action_spec.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ActionSpec:
    """
    行动规约：一个对工具调用的完整、独立的描述。
    这个数据类在整个系统中被复用，作为规划器和工具链之间传递“行动指令”的标准格式。
    """
    tool_name: str
    """要调用的工具的名称。"""

    parameters: Dict[str, Any]
    """调用该工具所需的参数。"""
