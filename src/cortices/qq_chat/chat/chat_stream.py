# src/cortices/qq_chat/chat_stream.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from src.common.event_model.event import Event
from src.common.event_model.info_data import ConversationInfo

# 定义滑动窗口大小常量
MAX_LLM_CONTEXT_SIZE = 12
MAX_RAW_EVENTS_SIZE = 12

class QQChatMessage(BaseModel):
    """
    代表一个标准化的聊天消息，用于构建 LLM 上下文。
    """
    user_nickname: Optional[str] = None
    user_cardname: Optional[str] = None
    user_id: Optional[str] = None
    content: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class QQChatStream(BaseModel):
    """
    代表一个独立的 QQ 聊天流（如一个群聊或私聊）。
    这个对象将被保存在 WorldModel 中，作为 Cortex 内部处理的主要数据单元。
    """
    stream_id: str = Field(..., description="唯一的流 ID，直接使用conversation_id")
    conversation_info:ConversationInfo
    
    # LLM 相关上下文
    llm_context: List[QQChatMessage] = Field(default_factory=list, description="用于 LLM 请求的滑动窗口消息列表")
    
    # 原始事件记录
    raw_events: List[Event] = Field(default_factory=list, description="此流中发生的原始事件记录，可用于回溯和调试")
    
    # 状态与元数据
    unread_count: int = 0
    unread_since: Optional[int] = None # 未读消息开始的时间戳
    last_event_timestamp: Optional[int] = None # 最近一个事件的时间戳 (int)

    class Config:
        arbitrary_types_allowed = True

    def add_event(self, event: Event):
        """
        向聊天流中添加一个新事件，并更新相关状态。
        """
        # 添加原始事件，并应用滑动窗口
        self.raw_events.append(event)
        if len(self.raw_events) > MAX_RAW_EVENTS_SIZE:
            self.raw_events = self.raw_events[-MAX_RAW_EVENTS_SIZE:]

        # 更新最近事件时间戳
        self.last_event_timestamp = event.time
        
        if event.event_type == "message":

            # 创建并添加标准化的聊天消息
            chat_message = QQChatMessage(
                user_nickname=event.user_info.user_nickname if event.user_info else None,
                user_cardname=event.user_info.user_cardname if event.user_info else None,
                user_id=event.user_info.user_id if event.user_info else None,
                content=event.event_data.LLM_plain_text,
                timestamp=event.time
            )
            self.llm_context.append(chat_message)
            
            # 应用 LLM 上下文的滑动窗口
            if len(self.llm_context) > MAX_LLM_CONTEXT_SIZE:
                self.llm_context = self.llm_context[-MAX_LLM_CONTEXT_SIZE:]
            
            # 更新未读计数和未读开始时间
            if self.unread_count == 0:
                self.unread_since = event.time # 从 0 变为 1 时记录时间
            self.unread_count += 1
            
    def mark_as_read(self):
        """
        将此聊天流标记为已读。
        """
        self.unread_count = 0
        self.unread_since = None # 清除未读开始时间

    def build_chat_history_for_llm(self, separator: str = "\n---\n") -> str:
        """
        根据 llm_context 列表构建一个用于 LLM 输入的格式化聊天历史字符串。

        Args:
            separator: 用于分隔每条消息的字符串。

        Returns:
            格式化后的聊天历史字符串。
        """
        history_lines = []
        
        # 使用 enumerate 遍历 llm_context，获取索引 i 和消息对象 msg
        for i, msg in enumerate(self.llm_context):
            
            # 确定发送者名称：优先使用名片，其次是昵称，最后使用用户 ID
            sender_cardname = msg.user_cardname or "未知用户"
            sender_nickname = msg.user_nickname or "未知用户"
            
            # 格式化时间戳（可选，但通常有助于 LLM 理解顺序）
            timestamp_str = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # 格式化消息内容：[序号] [时间] 发送者: 消息内容
            # 注意：i 是从 0 开始的索引
            #TODO: 以后去除id，LLM想要的话，提供nickname 从数据库查找
            line = (
                f"[{i+1}] [{timestamp_str}] {sender_cardname}({sender_nickname} id:{msg.user_id}): {msg.content or '[消息内容为空]'}"
            )
            history_lines.append(line)
        
        # 使用指定的分隔符连接所有消息行
        return separator.join(history_lines)