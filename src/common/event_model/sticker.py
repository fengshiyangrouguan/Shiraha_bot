from dataclasses import dataclass, field
from datetime import datetime
import uuid
from typing import List
from time import time
import os

@dataclass
class Sticker:
    """
    代表一个表情包的标准化数据模型。
    """
    full_path:str
    sticker_id: str
    path = os.path.dirname(full_path)  # 文件所在的目录路径
    filename = os.path.basename(full_path)  # 文件名
    embedding = []
    sticker_hash: str = field(default_factory=lambda: str(uuid.uuid4()))
    description = ""
    emotion: List[str] = []
    usage_count = 0
    last_used_time = time.time()
    register_time = time.time()
    format = ""

    def __hash__(self):
        # 允许将 Sticker 对象放入哈希集合（Set）中
        return hash(self.sticker_id)

    def __eq__(self, other):
        # 定义对象的相等性
        if not isinstance(other, Sticker):
            return NotImplemented
        return self.sticker_id == other.sticker_id
