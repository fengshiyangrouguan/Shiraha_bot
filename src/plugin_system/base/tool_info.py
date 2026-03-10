from enum import Enum
from typing import Dict, Any, List
from dataclasses import dataclass, field
from.parameter_info import ToolParameter


# 聊天模式枚举
class ChatMode(Enum):
    """聊天模式枚举"""

    FOCUS = "focus"  # Focus聊天模式
    NORMAL = "normal"  # Normal聊天模式
    PRIORITY = "priority"  # 优先级聊天模式
    ALL = "all"  # 所有聊天模式

    def __str__(self):
        return self.value


# 事件类型枚举
class EventType(Enum):
    """
    事件类型枚举类
    """

    ON_START = "on_start"  # 启动事件，用于调用按时任务
    ON_STOP = "on_stop"  # 停止事件，用于调用按时任务
    ON_MESSAGE = "on_message"
    ON_PLAN = "on_plan"
    POST_LLM = "post_llm"
    AFTER_LLM = "after_llm"
    POST_SEND = "post_send"
    AFTER_SEND = "after_send"
    UNKNOWN = "unknown"  # 未知事件类型

    def __str__(self) -> str:
        return self.value

@dataclass
class ToolInfo:
    """工具组件信息"""

    name: str  # 组件名称
    description: str = ""  # 组件描述
    enabled: bool = True  # 是否启用
    plugin_name: str = ""  # 所属插件名称
    is_built_in: bool = False  # 是否为内置组件
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    tool_parameters: List[ToolParameter] = field(default_factory=list)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


