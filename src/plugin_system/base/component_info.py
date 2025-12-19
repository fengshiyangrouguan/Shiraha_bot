from enum import Enum
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field


# 组件类型枚举
class ComponentType(Enum):
    """组件类型枚举"""

    ACTION = "action"  # 动作组件
    COMMAND = "command"  # 命令组件
    TOOL = "tool"  # 服务组件（预留）
    SCHEDULER = "scheduler"  # 定时任务组件（预留）
    EVENT_HANDLER = "event_handler"  # 事件处理组件（预留）

    def __str__(self) -> str:
        return self.value


# 动作激活类型枚举
class ActionActivationType(Enum):
    """动作激活类型枚举"""

    NEVER = "never"  # 从不激活（默认关闭）
    ALWAYS = "always"  # 默认参与到planner
    LLM_JUDGE = "llm_judge"  # LLM判定是否启动该action到planner
    RANDOM = "random"  # 随机启用action到planner
    KEYWORD = "keyword"  # 关键词触发启用action到planner

    def __str__(self):
        return self.value


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
class ComponentInfo:
    """组件信息"""

    name: str  # 组件名称
    component_type: ComponentType  # 组件类型
    description: str = ""  # 组件描述
    enabled: bool = True  # 是否启用
    plugin_name: str = ""  # 所属插件名称
    is_built_in: bool = False  # 是否为内置组件
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ActionInfo(ComponentInfo):
    """动作组件信息"""

    action_parameters: Dict[str, str] = field(
        default_factory=dict
    )  # 动作参数与描述，例如 {"param1": "描述1", "param2": "描述2"}
    action_require: List[str] = field(default_factory=list)  # 动作需求说明
    associated_types: List[str] = field(default_factory=list)  # 关联的消息类型
    # 激活类型相关
    activation_type: ActionActivationType = ActionActivationType.ALWAYS
    random_activation_probability: float = 0.0
    llm_judge_prompt: str = ""
    activation_keywords: List[str] = field(default_factory=list)  # 激活关键词列表
    keyword_case_sensitive: bool = False
    # 模式和并行设置
    parallel_action: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.activation_keywords is None:
            self.activation_keywords = []
        if self.action_parameters is None:
            self.action_parameters = {}
        if self.action_require is None:
            self.action_require = []
        if self.associated_types is None:
            self.associated_types = []
        self.component_type = ComponentType.ACTION


@dataclass
class CommandInfo(ComponentInfo):
    """命令组件信息"""

    command_pattern: str = ""  # 命令匹配模式（正则表达式）

    def __post_init__(self):
        super().__post_init__()
        self.component_type = ComponentType.COMMAND


@dataclass
class ToolInfo(ComponentInfo):
    """工具组件信息"""

    tool_parameters: List[Dict[str, Any]] = field(default_factory=list)  # 工具参数定义

    def __post_init__(self):
        super().__post_init__()
        self.component_type = ComponentType.TOOL


@dataclass
class EventHandlerInfo(ComponentInfo):
    """事件处理器组件信息"""

    event_type: EventType | str = EventType.ON_MESSAGE  # 监听事件类型
    intercept_message: bool = False  # 是否拦截消息处理（默认不拦截）
    weight: int = 0  # 事件处理器权重，决定执行顺序

    def __post_init__(self):
        super().__post_init__()
        self.component_type = ComponentType.EVENT_HANDLER
