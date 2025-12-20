# src/cortices/qq_chat/qq_data.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .chat_stream import QQChatStream
from src.common.event_model.info_data import ConversationInfo
from src.common.database.database_manager import DatabaseManager
from src.system.di.container import container

class QQChatData(BaseModel):
    """
    作为 qq_chat Cortex 在 WorldModel 中存储的顶层数据对象。
    它封装了与 QQ 聊天功能相关的所有状态。
    """
    streams: Dict[str, QQChatStream] = Field(default_factory=dict)
    bot_id: Optional[str] = None

    def get_or_create_stream(self, conversation_info:ConversationInfo) -> QQChatStream:
        """
        获取一个聊天流，如果不存在则创建并返回。
        """
        if conversation_info.conversation_id not in self.streams:
            self.streams[conversation_info.conversation_id] = QQChatStream(stream_id=conversation_info.conversation_id, conversation_info=conversation_info, bot_id=self.bot_id)
        return self.streams[conversation_info.conversation_id]

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

    def get_stream_by_id(self, stream_id: str) -> Optional[QQChatStream]:
        """
        通过 stream ID 获取一个已存在的 QQChatStream 实例。
        如果 Stream 不存在，则返回 None。
        """
        return self.streams.get(stream_id)
    
    def get_all_streams_history_for_llm(self) -> str:
        """
        获取所有聊天流的格式化历史记录。
        格式要求：每个Stream的历史记录前应有对话名称和ID的标题。
        
        Returns:
            包含所有格式化历史记录的单一字符串。
        """
        all_history_parts = []
        
        # 遍历存储的所有聊天流
        for stream_id, stream in self.streams.items():
            
            # 1. 提取元数据
            name = stream.conversation_info.conversation_name
            type = stream.conversation_info.conversation_type
            if type == "group":
                type_descripe = "群聊"
            else:
                type_descripe = "私聊"
            # 2. 构建 Stream 标题/分隔符
            header = (
                f"### {type_descripe}: {name}\n"
                f"conversation_id: {stream_id}\n"
                f"--- 聊天记录开始 ---\n"
            )
            
            # 3. 获取 Stream 内部的聊天历史
            # 调用 QQChatStream 中已有的方法来格式化消息列表
            chat_history = stream.build_chat_history_for_llm(separator="\n---\n")
            
            # 4. 组合并添加到列表中
            # 使用两个换行符 \n\n 来确保不同 Stream 之间有清晰的视觉分隔
            full_entry = f"{header}{chat_history}\n--- 聊天记录结束 ---\n\n"
            all_history_parts.append(full_entry)

        # 将所有 Stream 的历史记录连接成一个大字符串
        return "".join(all_history_parts).strip()