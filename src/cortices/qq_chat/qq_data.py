# src/cortices/qq_chat/qq_data.py
from typing import Dict, List
from pydantic import BaseModel, Field

from .chat_stream import QQChatStream

class QQChatData(BaseModel):
    """
    作为 qq_chat Cortex 在 WorldModel 中存储的顶层数据对象。
    它封装了与 QQ 聊天功能相关的所有状态。
    """
    streams: Dict[str, QQChatStream] = Field(default_factory=dict)

    def get_or_create_stream(self, stream_id: str) -> QQChatStream:
        """
        获取一个聊天流，如果不存在则创建并返回。
        """
        if stream_id not in self.streams:
            self.streams[stream_id] = QQChatStream(stream_id=stream_id)
        return self.streams[stream_id]

    @property
    def total_unread_count(self) -> int:
        """
        计算所有聊天流的未读消息总数。
        供 MotiveEngine 使用以产生宏观动机。
        """
        return sum(stream.unread_count for stream in self.streams.values())

    def get_unread_streams(self) -> List[QQChatStream]:
        """
        获取所有包含未读消息的聊天流列表。
        供 MainPlanner 使用以制定具体行动计划。
        """
        return [
            stream for stream in self.streams.values() if stream.unread_count > 0
        ]

    class Config:
        arbitrary_types_allowed = True
