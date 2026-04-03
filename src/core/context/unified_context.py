"""
Unified Context - 统一上下文系统

所有上下文采用 {role, content} 标准格式，支持自由拼接。
"""
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class Role(Enum):
    """角色类型"""
    SYSTEM = "system"           # 系统指令、人格设定
    USER = "user"               # 用户输入
    ASSISTANT = "assistant"     # Agent 输出
    TOOL = "tool"               # 工具执行结果
    MEMORY = "memory"           # 记忆检索结果
    OBSERVATION = "observation" # 观察/感知结果


@dataclass
class ContextMessage:
    """
    上下文消息

    统一的消息格式，支持各种类型的上下文信息
    """
    role: str                   # 角色
    content: str                # 内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    # 可选的时间戳
    timestamp: Optional[float] = None

    # 可选的关联任务ID
    task_id: Optional[str] = None

    # 可选的源信息
    source_cortex: Optional[str] = None
    source_target: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（LLM 标准格式）"""
        d = {"role": self.role, "content": self.content}
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def is_system(self) -> bool:
        return self.role == Role.SYSTEM.value

    def is_user(self) -> bool:
        return self.role == Role.USER.value

    def is_assistant(self) -> bool:
        return self.role == Role.ASSISTANT.value

    def is_tool(self) -> bool:
        return self.role == Role.TOOL.value

    def is_memory(self) -> bool:
        return self.role == Role.MEMORY.value

    def is_observation(self) -> bool:
        return self.role == Role.OBSERVATION.value

    def __str__(self) -> str:
        return f"[{self.role}] {self.content[:50]}..."


class UnifiedContext:
    """
    统一上下文

    管理消息列表，提供构建、拼接、过滤等操作
    """

    def __init__(self):
        self.messages: List[ContextMessage] = []
        self._max_tokens = 4096  # 最大 token 数

    def append(self, message: Union[ContextMessage, Dict[str, str], str], **kwargs) -> "UnifiedContext":
        """
        添加消息

        Args:
            message: 消息，可以是 ContextMessage、dict 或字符串
            **kwargs: 额外元数据

        Returns:
            self，支持链式调用
        """
        if isinstance(message, ContextMessage):
            self.messages.append(message)
        elif isinstance(message, dict):
            self.messages.append(ContextMessage(
                role=message.get("role", "user"),
                content=message.get("content", ""),
                metadata=message.get("metadata", {}),
                **kwargs
            ))
        elif isinstance(message, str):
            # 默认为 user 消息
            self.messages.append(ContextMessage(
                role=Role.USER.value,
                content=message,
                **kwargs
            ))
        return self

    def prepend(self, message: Union[ContextMessage, Dict[str, str], str], **kwargs) -> "UnifiedContext":
        """在开头添加消息"""
        if isinstance(message, ContextMessage):
            self.messages.insert(0, message)
        elif isinstance(message, dict):
            self.messages.insert(0, ContextMessage(
                role=message.get("role", "user"),
                content=message.get("content", ""),
                metadata=message.get("metadata", {}),
                **kwargs
            ))
        elif isinstance(message, str):
            self.messages.insert(0, ContextMessage(
                role=Role.USER.value,
                content=message,
                **kwargs
            ))
        return self

    def extend(self, other: "UnifiedContext") -> "UnifiedContext":
        """
        合并另一个上下文

        Args:
            other: 另一个 UnifiedContext

        Returns:
            self，支持链式调用
        """
        self.messages.extend(other.messages)
        return self

    def filter_by_role(self, role: str) -> List[ContextMessage]:
        """按角色过滤"""
        role_value = role if isinstance(role, str) else role.value
        return [m for m in self.messages if m.role == role_value]

    def filter_by_source(self, source_cortex: str) -> List[ContextMessage]:
        """按来源域过滤"""
        return [m for m in self.messages if m.source_cortex == source_cortex]

    def filter_by_target(self, source_target: str) -> List[ContextMessage]:
        """按来源目标过滤"""
        return [m for m in self.messages if m.source_target == source_target]

    def filter_by_task(self, task_id: str) -> List[ContextMessage]:
        """按任务ID过滤"""
        return [m for m in self.messages if m.task_id == task_id]

    def get_recent(self, n: int = 5) -> List[ContextMessage]:
        """获取最近的 n 条消息"""
        return self.messages[-n:] if len(self.messages) >= n else list(self.messages)

    def get_by_role_sequence(self, roles: List[str]) -> List[ContextMessage]:
        """
        按角色顺序获取最近的消息序列

        例如：获取最近的 user -> assistant -> user 序列
        """
        result = []
        role_idx = len(roles) - 1

        # 从后往前扫描
        for message in reversed(self.messages):
            if message.role == roles[role_idx]:
                result.append(message)
                role_idx -= 1
                if role_idx < 0:
                    break

        return list(reversed(result))

    def count_tokens(self) -> int:
        """估算 token 数量（粗略估算：中文字符 * 0.7 + 英文字符）"""
        total = 0
        for message in self.messages:
            # 简单估算
            chinese_chars = sum(1 for c in message.content if '\u4e00' <= c <= '\u9fff')
            other_chars = len(message.content) - chinese_chars
            total += chinese_chars + other_chars // 4  # 英文大约 4 字符 = 1 token
        return total

    def truncate(self, max_tokens: Optional[int] = None) -> "UnifiedContext":
        """
        截断上下文以适应 token 限制

        保留系统消息，从旧到新截断

        Args:
            max_tokens: 最大 token 数，默认使用 _max_tokens

        Returns:
            self，支持链式调用
        """
        limit = max_tokens or self._max_tokens

        # 先保留系统消息
        system_messages = self.filter_by_role(Role.SYSTEM.value)
        non_system_messages = [m for m in self.messages if not m.is_system()]

        # 计算系统消息的 token 数
        system_tokens = sum(len(m.content) for m in system_messages)
        remaining_tokens = limit - system_tokens

        # 截断非系统消息
        truncated_messages = []
        current_tokens = 0

        # 从最新消息开始，往前添加
        for message in reversed(non_system_messages):
            msg_tokens = len(message.content)
            if current_tokens + msg_tokens <= remaining_tokens:
                truncated_messages.insert(0, message)
                current_tokens += msg_tokens
            else:
                break

        # 重建消息列表（系统消息在前）
        self.messages = system_messages + truncated_messages

        return self

    def clear(self) -> "UnifiedContext":
        """清空所有消息"""
        self.messages.clear()
        return self

    def to_list(self) -> List[Dict[str, str]]:
        """
        转换为标准消息列表

        用于 LLM 调用
        """
        return [m.to_dict() for m in self.messages]

    def clone(self) -> "UnifiedContext":
        """克隆上下文"""
        new_context = UnifiedContext()
        new_context.messages = [ContextMessage(
            role=m.role,
            content=m.content,
            metadata=m.metadata.copy(),
            timestamp=m.timestamp,
            task_id=m.task_id,
            source_cortex=m.source_cortex,
            source_target=m.source_target
        ) for m in self.messages]
        new_context._max_tokens = self._max_tokens
        return new_context

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"UnifiedContext(messages={len(self.messages)}, tokens≈{self.count_tokens()})"

    def __str__(self) -> str:
        parts = []
        for msg in self.messages:
            parts.append(f"[{msg.role}] {msg.content[:30]}...")
        return "\n".join(parts)
