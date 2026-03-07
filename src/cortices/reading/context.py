# src/cortices/reading/context.py
from collections import deque
from typing import List, Deque
from pydantic import BaseModel, Field

class AgentReadingContext(BaseModel):
    """
    用于在内存中维护 Agent 自己的阅读任务状态。
    这个上下文是面向 Agent 内部的，而非外部用户。
    """
    book_id: int
    book_title: str
    
    # 书籍的全部内容，已切分成片段
    chunks: List[str]
    current_chunk_index: int = 0
    
    # 使用 deque 作为 Agent 的短期记忆，存储最近的读后感
    short_term_memory: Deque[str] = Field(default_factory=lambda: deque(maxlen=10))

    @property
    def total_chunks(self) -> int:
        """计算总片段数。"""
        return len(self.chunks)

    def get_current_chunk(self) -> str:
        """获取当前要阅读的片段。"""
        if 0 <= self.current_chunk_index < self.total_chunks:
            return self.chunks[self.current_chunk_index]
        return None # 返回 None 表示已读完

    def advance_to_next_chunk(self):
        """将索引推进到下一个片段。"""
        self.current_chunk_index += 1
