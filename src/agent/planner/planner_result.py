from dataclasses import dataclass
from typing import Dict, Any
@dataclass
class PlanResult:
    """planner 规划结果的标准化数据结构。"""
    thought: str
    tool_name: str
    parameters: Dict[str, Any]