# src/cortices/qq_chat/chat_stream.py
import asyncio
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from src.common.logger import get_logger
from src.common.event_model.event import Event
from src.common.event_model.info_data import ConversationInfo

from src.common.di.container import container
from src.common.database.database_manager import DatabaseManager
from src.agent.world_model import WorldModel
from src.common.database.database_model import UserInfoDB, ConversationInfoDB, EventDB

# 定义滑动窗口大小常量
MAX_LLM_CONTEXT_SIZE = 30
MAX_RAW_EVENTS_SIZE = 30

logger = get_logger("ChatStream")

class QQChatMessage(BaseModel):
    """
    代表一个标准化的聊天消息，用于构建 LLM 上下文。
    """
    message_id: Optional[str] = None
    user_nickname: Optional[str] = None
    user_cardname: Optional[str] = None
    user_id: Optional[str] = None
    content: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = Field(default_factory=list)
    is_replyed:bool = False

class QQChatStream(BaseModel):
    """
    代表一个独立的 QQ 聊天流（如一个群聊或私聊）。
    这个对象将被保存在 WorldModel 中，作为 Cortex 内部处理的主要数据单元。
    """
    database_manager:DatabaseManager = container.resolve(DatabaseManager)
    world_model:WorldModel = container.resolve(WorldModel)
    stream_id: str = Field(..., description="唯一的流 ID，直接使用conversation_id")
    bot_id: Optional[str] = None
    conversation_info:ConversationInfo

    # asyncio.Event 实例，作为收到新消息的信号旗，用于更新深度聊天的planner
    _new_message_event = asyncio.Event()

    # 在群聊中，每积累n条消息才重开始规划，防止token大量消耗
    _new_plan_msg_threshold_for_group:int = 5
    _new_plan_msg_threshold_for_private:int = 1
    _msg_count:int = 0
    
    # LLM 相关上下文
    llm_context: List[QQChatMessage] = Field(default_factory=list, description="用于 LLM 请求的滑动窗口消息列表")
    
    # 原始事件记录
    raw_events: List[Event] = Field(default_factory=list, description="此流中发生的原始事件记录，可用于回溯和调试")
    
    # 状态与元数据
    unread_count: int = 0
    last_event_timestamp: Optional[int] = None # 最近一个事件的时间戳 (int)

    @property
    def unreplied_ats(self) -> List[QQChatMessage]:
        """获取所有未回复且明确 @ 了 Bot 的消息列表 (硬中断)"""
        return [
            msg for msg in self.llm_context 
            if not msg.is_replyed and ("at_me" in msg.tags)
        ]

    @property
    def unreplied_mentions(self) -> List[QQChatMessage]:
        """获取所有未回复且提及了名字（但没 @）的消息列表 (软感知)"""
        return [
            msg for msg in self.llm_context 
            if not msg.is_replyed and "mentioned_me" in msg.tags
        ]
        
    class Config:
        arbitrary_types_allowed = True
        
    async def add_event(self, event: Event):
        """
        向聊天流中添加一个新事件，并更新相关状态。
        """
        # 添加原始事件，并应用滑动窗口
        self._msg_count += 1

        self.raw_events.append(event)
        if len(self.raw_events) > MAX_RAW_EVENTS_SIZE:
            self.raw_events = self.raw_events[-MAX_RAW_EVENTS_SIZE:]

        # 更新最近事件时间戳
        self.last_event_timestamp = event.time
                
        user_db: Optional[UserInfoDB] = None
        if event.user_info:
            user_db = UserInfoDB(
                user_id=event.user_info.user_id,
                user_nickname=event.user_info.user_nickname,
                user_cardname=event.user_info.user_cardname
            )
            await self.database_manager.upsert(user_db)
            logger.debug(f"UserInfoDB (ID: {user_db.user_id}) 已保存。")

        conversation_db: Optional[ConversationInfoDB] = None
        if event.conversation_info:
            conversation_db = ConversationInfoDB(
                conversation_id=event.conversation_info.conversation_id,
                conversation_type=event.conversation_info.conversation_type,
                conversation_name=event.conversation_info.conversation_name,
                parent_id=event.conversation_info.parent_id,
                platform_meta=event.conversation_info.platform_meta
            )
            await self.database_manager.upsert(conversation_db)

            logger.debug(f"ConversationInfoDB (ID: {conversation_db.conversation_id}) 已保存。")
        
        event_content = event.event_data.LLM_plain_text
        event_tags_list = list(event.tags) if event.tags else []

        event_db = EventDB(
            event_id=event.event_id,
            platform=event.platform,
            event_type=event.event_type,
            time=event.time,
            conversation_id=event.conversation_info.conversation_id if event.conversation_info else None,
            conversation_type=event.conversation_info.conversation_type if event.conversation_info else None,
            conversation_name=event.conversation_info.conversation_name if event.conversation_info else None,
            user_id=event.user_info.user_id if event.user_info else None,
            user_nickname=event.user_info.user_nickname if event.user_info else None,
            user_cardname=event.user_info.user_cardname if event.user_info else None,
            tags=event_tags_list,
            event_content=event_content, # 存储为 JSON 字符串
            event_metadata=event.event_data.metadata
        )
        await self.database_manager.upsert(event_db)

        #TODO:暂时删掉了群聊和私聊的不同规划周期，因为插件系统的的跑团问题
        if self._msg_count > self._new_plan_msg_threshold_for_private:
            self._new_message_event.set()
            self._msg_count = 0

        logger.debug(f"EventDB (ID: {event_db.event_id}) 已永久化存储。")
        logger.debug(f"事件 ({event.event_type}) 针对流 ({self.stream_id}) 处理完成。")


        if event.event_type == "message":

            # 创建并添加标准化的聊天消息
            chat_message = QQChatMessage(
                message_id=event.event_id,
                user_nickname=event.user_info.user_nickname if event.user_info else None,
                user_cardname=event.user_info.user_cardname if event.user_info else None,
                user_id=event.user_info.user_id if event.user_info else None,
                content=event.event_data.LLM_plain_text,
                timestamp=event.time,
                tags=list(event.tags) if event.tags else []
            )
            self.llm_context.append(chat_message)
            
            # 应用 LLM 上下文的滑动窗口
            if len(self.llm_context) > MAX_LLM_CONTEXT_SIZE:
                self.llm_context = self.llm_context[-MAX_LLM_CONTEXT_SIZE:]
            
            # 更新未读计数
            if "self_message" not in event.tags:
                self.unread_count += 1
            
    def mark_as_read(self):
        """
        将此聊天流标记为已读。
        """
        self.unread_count = 0

    def build_chat_history_for_summary(self, separator: str = "\n") -> str:
        """
        根据 llm_context 列表构建一个用于 LLM 输入的格式化聊天历史字符串。

        Args:
            separator: 用于分隔每条消息的字符串。

        Returns:
            格式化后的聊天历史字符串。
        """
        history_lines = []
        
        if not self.llm_context:
            return "无最近聊天记录"
        
        # 使用 enumerate 遍历 llm_context，获取索引 i 和消息对象 msg
        for i, msg in enumerate(self.llm_context):
            
            # 确定发送者名称：优先使用名片，其次是昵称，最后使用用户 ID
            sender_nickname = msg.user_nickname or "未知用户"
            
            # 格式化时间戳（可选，但通常有助于 LLM 理解顺序）
            timestamp_str = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")

            if self.bot_id is not None and str(msg.user_id) == str(self.bot_id):
                sender_nickname = "你自己"

            # 格式化消息内容：[序号] [时间] 发送者: 消息内容
            # 注意：i 是从 0 开始的索引
            #TODO: 以后去除id，LLM想要的话，提供nickname 从数据库查找
            line = (
                # f"[{i+1}] [{timestamp_str}] {sender_cardname}({sender_nickname}): {msg.content or '[消息内容为空]'}"
                f"[{timestamp_str}] {sender_nickname}: {msg.content or '[消息内容为空]'}"
            )
            history_lines.append(line)
        
        # 使用指定的分隔符连接所有消息行
        return separator.join(history_lines)

    def build_chat_history_has_msg_id(self, separator: str = "\n") -> str:
        """
        根据 llm_context 列表构建一个用于 LLM 输入的格式化聊天历史字符串,且含有消息 ID。

        Args:
            separator: 用于分隔每条消息的字符串。

        Returns:
            格式化后的聊天历史字符串。
        """
        history_lines = []
        divider_inserted = False
        
        if not self.llm_context:
            return "无最近聊天记录"
        
        # 使用 enumerate 遍历 llm_context，获取索引 i 和消息对象 msg
        for i, msg in enumerate(self.llm_context):
            if not msg.is_replyed and not divider_inserted:
                if history_lines:  # 确保前面有历史消息才加分割线
                    history_lines.append("—— 以上为已回复历史消息，禁止回复 ——")
                divider_inserted = True

            # 确定发送者名称：优先使用名片，其次是昵称，最后使用用户 ID
            sender_nickname = msg.user_nickname or "未知用户"
            msg_id = msg.message_id or "未知消息ID"
            # 格式化时间戳（可选，但通常有助于 LLM 理解顺序）
            timestamp_str = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")

            if self.bot_id is not None and str(msg.user_id) == str(self.bot_id):
                sender_nickname = "你自己"
                
            line = (
                f"[{timestamp_str}] {sender_nickname}(消息ID:{msg_id}): {msg.content or '[消息内容为空]'}"
            )
            history_lines.append(line)

        # 如果循环结束了依然没有插入过分割线，且历史记录不为空
        # 说明所有消息都是已读的，直接在最后追加标识符
        if not divider_inserted and history_lines:
            history_lines.append("—— 以上为已回复的历史消息，禁止回复 ——")
        
        # 使用指定的分隔符连接所有消息行
        return separator.join(history_lines)
    
    def build_openai_chat_history(self) -> List[Dict[str, str]]:
        """
        将 llm_context 转换为 OpenAI 标准消息格式，并处理已回复状态。
        """
        messages = []
        divider_inserted = False
        
        if not self.llm_context:
            return [{"role": "system", "content": "无最近聊天记录"}]

        for msg in self.llm_context:
            # 1. 处理分割线 (逻辑维持原样，但作为 system 消息插入)
            if not msg.is_replyed and not divider_inserted:
                if messages:
                    messages.append({
                        "role": "system", 
                        "content": "—— 以上为已回复历史消息，禁止重复回复 ——"
                    })
                divider_inserted = True

            # 2. 确定角色 (Role)
            is_bot = self.bot_id is not None and str(msg.user_id) == str(self.bot_id)
            role = "assistant" if is_bot else "user"

            # 3. 构造内容 (Content)
            # 对于非机器人消息，保留昵称和消息 ID 方便模型引用
            sender_name = "你自己" if is_bot else (msg.user_nickname or "未知用户")
            msg_id = msg.message_id or "未知ID"
        
            content = f"[{sender_name} (ID:{msg_id})]: {msg.content or '[空消息]'}"
            
            messages.append({
                "role": role,
                "content": content
            })

        # 4. 如果全都是已读，末尾补一个提示
        if not divider_inserted and messages:
            messages.append({
                "role": "system", 
                "content": "—— 以上为已回复的历史消息，禁止重复回复 ——"
            })

        return messages
    
    def mark_as_replyed(self):
        for i, msg in enumerate(self.llm_context):
            msg.is_replyed = True
            self._msg_count = 0
            self._new_message_event.clear() 

    def get_new_message_event(self) -> asyncio.Event:
        return self._new_message_event