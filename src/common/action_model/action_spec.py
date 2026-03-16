from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ActionSpec:
    """
    统一动作定义。

    当前默认 action_type 为 tool，后续也可以扩展到 command / control 等其他动作类型。
    """

    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    action_type: str = "tool"
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.tool_name

    @name.setter
    def name(self, value: str) -> None:
        self.tool_name = value
