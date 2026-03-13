# src/cortices/qq_chat/qq_data.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .chat_stream import QQChatStream
from src.common.event_model.info_data import ConversationInfo
from src.common.database.database_manager import DatabaseManager
from src.common.database.database_model import ConversationInfoDB
from src.common.di.container import container

class QQChatData(BaseModel):
    """
    作为 qq_chat Cortex 在 WorldModel 中存储的顶层数据对象。
    它封装了与 QQ 聊天功能相关的所有状态。
    """
    streams: Dict[str, QQChatStream] = Field(default_factory=dict)
    bot_id: Optional[str] = None
    

    def _get_or_create_stream(self, conversation_info:ConversationInfo) -> QQChatStream:
        """
        获取一个聊天流，如果不存在则创建并返回。
        """
        if conversation_info.conversation_id not in self.streams:
            self.streams[conversation_info.conversation_id] = QQChatStream(stream_id=conversation_info.conversation_id, conversation_info=conversation_info, bot_id=self.bot_id)
        return self.streams[conversation_info.conversation_id]
    
    async def get_or_create_stream_by_id(self, conversation_id: str) -> Optional[QQChatStream]:
        """
        核心逻辑：先从内存找，找不到则去数据库搜，搜到了就同步到内存，搜不到返回 None。
        """
        # 第一步：检查内存中是否已存在
        if conversation_id in self.streams:
            return self.streams[conversation_id]

        # 第二步：尝试从数据库兜底 (假设 ConversationInfoDB 是你的数据库模型)
        db_manager = container.resolve(DatabaseManager)
        conv_db = await db_manager.get(ConversationInfoDB, conversation_id)
        if conv_db:
            # 构造临时 Info 对象
            conversation_info = ConversationInfo(
                conversation_id=conversation_id,
                conversation_type=getattr(conv_db, "conversation_type", "private") or "private",
                conversation_name=getattr(conv_db, "conversation_name", "未知对象") or "未知对象"
            )
            # 使用同步方法同步到内存并返回
            return self._get_or_create_stream(conversation_info)
        
        return None

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