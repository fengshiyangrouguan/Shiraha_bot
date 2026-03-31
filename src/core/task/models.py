from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
import time
import uuid

class TaskStatus(Enum):
    READY = "ready"         # 就绪，等待调度
    FOCUS = "focus"         # 当前注意力位置
    SUSPENDED = "suspended" # 已挂起（保存了现场，等待恢复）
    BLOCKED = "blocked"     # 阻塞（如：等待搜索结果返回）
    BACKGROUND = "background" # 后台感知模式，不占用主逻辑注意力
    TERMINATED = "terminated" # 已结束

@dataclass
class TaskInstance:
    # 基础信息
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    cortex: str = ""           # 负责该任务的驱动 (如: qq_cortex)
    target_id: str = ""        # 交互目标 (如: 群号或用户ID)
    
    # 调度权重
    priority: int = 50         # 0-100
    
    # 状态控制
    status: TaskStatus = TaskStatus.READY
    
    # 上下文指针 (Context Ref)
    # 对应数据库中的 session_id 或 内存中的句柄
    context_ref: str = ""
    
    # 任务元数据 (用于 Summary 或 调试)
    motive: str = ""           # 初始意图
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # 扩展槽位 (用于存放子规划器返回的临时信号)
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in self.__dict__.items()}