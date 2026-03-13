# src/agent/planner/planner_result.py
from dataclasses import dataclass
from src.common.action_model.action_spec import ActionSpec

@dataclass
class PlanResult:
    """
    planner 规划结果的标准化数据结构。
    这是“规划”阶段的输出，作为“行动”阶段的输入。
    """
    reason: str
    """规划过程中的思考、推理或决策依据。"""

    action: ActionSpec
    """根据思考最终决定的、要执行的具体行动。"""
