import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


class Priority(Enum):
    """
    任务调度优先级。
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskMode(Enum):
    """
    Task 运行模式。

    新架构下，Task 不再承担“多态 action 栈”的语义，
    而是一个可持续存在的任务容器。
    """

    ONCE = "once"
    LISTEN = "listen"
    LOOP = "loop"
    CRON = "cron"


class TaskStatus(Enum):
    """
    任务生命周期状态机。
    """

    READY = "ready"
    FOCUS = "focus"
    SUSPENDED = "suspended"
    BLOCKED = "blocked"
    MUTED = "muted"
    BACKGROUND = "background"
    TERMINATED = "terminated"


class BaseAction(ABC):
    """
    兼容壳。

    该基类保留只是为了兼容旧代码的 import，
    新主链已经不再依赖多态 Action 作为正式运行模型。
    """

    def __init__(self, action_id: str, priority: Priority = Priority.LOW):
        self.action_id = action_id
        self.priority = priority
        self.is_completed = False
        self.result: Any = None
        self.is_suspended = False

    @abstractmethod
    async def execute(self, cortex: Any, context: Dict) -> Optional[str]:
        pass

    def on_perception(self, data: Any):
        pass

    def finalize(self, result: Any):
        self.is_completed = True
        self.result = result


@dataclass
class Task:
    """
    任务实体。

    新版本语义：
    1. Task 是调度和上下文归属的最小长期单元。
    2. 行为链不再由内部 action 栈驱动，而是由 Planner 按 mode 继续推进。
    3. `task_window` 用于收纳当前任务窗口的渐进展开上下文。
    """

    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    target_id: str = ""
    cortex: str = ""
    priority: Priority = Priority.LOW
    status: TaskStatus = TaskStatus.BACKGROUND
    mode: TaskMode = TaskMode.ONCE
    context_ref: str = ""
    motive: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # 新主链核心数据：
    task_window: List[Dict[str, Any]] = field(default_factory=list)
    task_config: Dict[str, Any] = field(default_factory=dict)
    last_signal: Dict[str, Any] = field(default_factory=dict)
    last_observation: str = ""
    last_result: Dict[str, Any] = field(default_factory=dict)
    view_cache: Dict[str, Any] = field(default_factory=dict)
    anchors: Dict[str, str] = field(default_factory=dict)
    execution_count: int = 0

    # 兼容旧代码读取，默认保留但不再作为主链正式语义。
    actions: List[BaseAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        输出可序列化字典。
        """
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif key == "actions":
                result[key] = [getattr(action, "action_id", str(action)) for action in value]
            else:
                result[key] = value
        return result

    def add_action(self, action: BaseAction):
        """
        兼容旧接口。

        新架构不会继续扩展这条路径，但短期内保留，避免旧代码 import 后直接崩溃。
        """
        self.actions.append(action)

    def append_window_message(self, role: str, content: str, **metadata) -> None:
        """
        向任务窗口追加一条统一格式消息。
        """
        message = {
            "role": role,
            "content": content,
        }
        if metadata:
            message["metadata"] = metadata
        self.task_window.append(message)
        self.updated_at = time.time()

    def set_last_signal(self, signal_payload: Dict[str, Any]) -> None:
        """更新最近一次唤醒/中断信号。"""
        self.last_signal = signal_payload
        self.updated_at = time.time()

    def set_last_result(self, result_payload: Dict[str, Any]) -> None:
        """更新最近一次执行结果。"""
        self.last_result = result_payload
        self.updated_at = time.time()

    def cache_view(self, panel_name: str, panel_data: Any) -> None:
        """缓存显式查看过的控制面板结果。"""
        self.view_cache[panel_name] = panel_data
        self.updated_at = time.time()
