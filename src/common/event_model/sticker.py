from dataclasses import dataclass, field
from datetime import datetime
import uuid
from typing import List
import time
import os

@dataclass
class Sticker:
    """
    代表一个表情包的标准化数据模型。
    """
    file_path:str
    embedding: List[float] = field(default_factory=list)
    sticker_hash: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    emotion: List[str] = field(default_factory=list)
    usage_count: int = 0
    last_used_time: float = field(default_factory=time.time)
    register_time: float = field(default_factory=time.time)
    file_format: str = ""

    def __hash__(self):
        # 允许将 Sticker 对象放入哈希集合（Set）中
        return hash(self.sticker_hash)

    def __eq__(self, other):
        # 定义对象的相等性
        if not isinstance(other, Sticker):
            return NotImplemented
        return self.sticker_hash == other.sticker_hash
