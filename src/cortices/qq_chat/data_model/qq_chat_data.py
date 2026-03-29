# src/cortices/qq_chat/qq_data.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .chat_stream import QQChatStream, QQChatMessage
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
    

    def get_or_create_stream(self, conversation_info:ConversationInfo) -> QQChatStream:
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
            return self.get_or_create_stream(conversation_info)
        
        return None

    @property
    def total_unread_count(self) -> int:
        """
        计算所有聊天流的未读消息总数。
        供 MotiveEngine 使用以产生宏观动机。
        """
        return sum(stream.unread_count for stream in self.streams.values())
    

    async def get_global_perception_report(self) -> str:
        """
        生成结构化的 QQ 消息列表概览。
        仅在有特定社交触发时显式标注 [有人@你] 或 [有人提及你]。
        """
        db_manager = container.resolve(DatabaseManager)
        from sqlmodel import select, desc
        from src.common.database.database_model import ConversationInfoDB
        
        # 建议增加排序，让最近活跃的排在前面，方便 Planner 聚焦
        all_convs: List[ConversationInfoDB] = await db_manager.get_all(select(ConversationInfoDB))
        
        if not all_convs:
            return "QQ 会话列表\n(暂无通讯记录)"

        report = [" QQ 会话列表概览"]

        for conv in all_convs:
            s_id = str(conv.conversation_id)
            stream = self.streams.get(s_id)
            
            unread_count = stream.unread_count if stream else 0
            type_str = "群聊" if conv.conversation_type == "group" else "私聊"
            
            # 社交标签构建
            social_tags = ""
            detail_lines = []
            
            if stream:
                # 处理 @ 消息
                ats = stream.unreplied_ats
                m:QQChatMessage = None
                if ats:
                    social_tags += "[有人@你]"
                    for m in ats:
                        detail_lines.append(f"  [@]{m.user_nickname}:{m.content or '[消息内容为空]'}")
                
                # 处理 提及 消息
                mentions = stream.unreplied_mentions
                if mentions:
                    social_tags += " [有人提及你]"
                    for m in mentions:
                        detail_lines.append(f"  [提及]{m.user_nickname}:{m.content or '[消息内容为空]'}")

            # 组装单行信息
            unread_info = f"({unread_count}条未读消息)" if unread_count > 0 else "(所有消息均已读)"
            
            # 格式：- [群聊] 开发测试群 (ID: 12345) (3条未读) [有人@你]
            line = f"- [{type_str}] {conv.conversation_name} (ID: {s_id}) {unread_info}{social_tags}"
            report.append(line)
            
            # 注入详细内容预览
            if detail_lines:
                report.extend(detail_lines)

        return "\n".join(report)