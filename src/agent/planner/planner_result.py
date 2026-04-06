from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PlannerResult:
    """
    新版 Planner 统一输出。

    主 Planner 与回复规划器都应尽量收敛到这个结构：
    1. `thought` 用于表达面对最新输入时的想法、动机或判断。
    2. `shell_commands` 是真正要执行的内核指令列表。
    3. `raw_content` 保留原始输出，便于调试。
    """

    thought: str = ""
    shell_commands: List[str] = field(default_factory=list)
    raw_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "shell_commands": list(self.shell_commands),
            "raw_content": self.raw_content,
            "metadata": dict(self.metadata),
        }


# 兼容旧命名，避免旧模块导入直接失败。
PlanResult = PlannerResult
