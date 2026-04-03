"""
Memory Entry - 记忆条目模型

定义记忆的基本结构，支持混合存储的不同层次。
"""
import time
import uuid
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field


class MemoryType(Enum):
    """记忆类型"""
    WORKING = "working"      # 工作记忆，当前关注
    SHORT_TERM = "short_term"  # 短期记忆，最近行为
    LONG_TERM = "long_term"    # 长期记忆，向量化存储


@dataclass
class MemoryEntry:
    """
    记忆条目

    支持跨域、跨时间的记忆检索和引用
    """
    memory_id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    content: str = ""                    # 原始内容
    embedding: Optional[List[float]] = None  # 向量（用于语义检索）
    memory_type: MemoryType = MemoryType.SHORT_TERM  # 记忆类型

    # 来源信息
    source_cortex: str = ""              # 来源域（qq/reading/browser/...）
    source_target: str = ""              # 来源归属（用户ID/群ID/...")
    source_action: Optional[str] = None  # 产生此记忆的行为

    # 时间信息
    timestamp: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # 标记信息
    tags: List[str] = field(default_factory=list)  # 标签
    keywords: List[str] = field(default_factory=list)  # 关键词
    importance: float = 0.5             # 重要性（0-1）
    access_count: int = 0               # 访问次数

    # 关联信息
    related_task_id: Optional[str] = None  # 关联的任务ID
    related_context: Optional[Dict[str, Any]] = None  # 关联的上下文信息

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "embedding": self.embedding,
            "memory_type": self.memory_type.value,
            "source_cortex": self.source_cortex,
            "source_target": self.source_target,
            "source_action": self.source_action,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "tags": self.tags,
            "keywords": self.keywords,
            "importance": self.importance,
            "access_count": self.access_count,
            "related_task_id": self.related_task_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """从字典创建实例"""
        return cls(
            memory_id=data.get("memory_id", f"mem_{uuid.uuid4().hex[:12]}"),
            content=data.get("content", ""),
            embedding=data.get("embedding"),
            memory_type=MemoryType(data.get("memory_type", "short_term")),
            source_cortex=data.get("source_cortex", ""),
            source_target=data.get("source_target", ""),
            source_action=data.get("source_action"),
            timestamp=data.get("timestamp", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            tags=data.get("tags", []),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            related_task_id=data.get("related_task_id"),
        )

    def mark_accessed(self):
        """标记为已访问"""
        self.last_accessed = time.time()
        self.access_count += 1

    def is_expired(self, ttl_seconds: float) -> bool:
        """检查是否过期"""
        if self.memory_type == MemoryType.WORKING:
            # 工作记忆过期时间短
            ttl = ttl_seconds * 0.1
        elif self.memory_type == MemoryType.SHORT_TERM:
            ttl = ttl_seconds * 0.5
        else:
            # 长期记忆不过期
            return False
        return time.time() - self.timestamp > ttl

    def __repr__(self) -> str:
        return f"MemoryEntry(id={self.memory_id[:8]}, type={self.memory_type.value}, content={self.content[:30]}...)"
