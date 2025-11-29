from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set
import copy

from src.plugin_system.event_types import EventType # 导入事件类型
from src.common.event_model.info_data import ConversationInfo, UserInfo
from src.common.event_model.event_data import BaseEventData

@dataclass
class Event:
    """
    Shiraha_bot 内部使用的事件对象。
    这个对象将在消息处理管道中传递和修改。
    """

    # --- 事件基础字段 ---
    event_type: str             # message, notice, system, reaction……
    event_id: str
    time: int
    
    # --- 平台与来源信息 ---
    platform: str               # qq, wechat, discord, telegram……
    chat_stream_id: str = None
    user_info: Optional[UserInfo] = None
    conversation_info: Optional[ConversationInfo] = None

    # --- 可扩展 tag  ---
    tags: Set[str] = field(default_factory=set)

    # --- 事件主内容（MessageEvent / NoticeEvent / etc） ---
    event_data: Optional[BaseEventData] = None

    # --- Metadata：插件、管道可写入的通用字段 ---
    metadata: Dict[str, Any] = field(default_factory=dict)

     # --- 事件内部处理控制 ---
    _propagation_stopped: bool = False   # 插件设置该标志位，阻断传播，默认不阻断

    # 外部依赖（事件行为方法 需要 注入 manager）
    #_platform_manager = field(default=None, repr=False, compare=False)
    #_event_manager = field(default=None, repr=False, compare=False)


    # ----------------------------------------------------------------------
    # ------------------------ 基础方法（通用）------------------------------
    # ----------------------------------------------------------------------

    # Tag 系统
    def add_tag(self, tag: str):
        """
        给事件打新标签
        """
        self.tags.add(tag)

    def has_tag(self, tag: str) -> bool:
        """
        检查是否含有该tag
        """
        return tag in self.tags

    # Metadata
    def set_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None):
        return self.metadata.get(key, default)

    # 停止事件传播（给 pipeline / plugin 用）
    def stop_propagation(self):
        self._propagation_stopped = True

    def is_stopped(self) -> bool:
        return self._propagation_stopped

    # 创建事件的深度副本
    def copy(self) -> "Event":
        return copy.deepcopy(self)

    # ----------------------------------------------------------------------
    # ------------------------ 事件行为（调用 PlatformManager）--------------
    # ----------------------------------------------------------------------

    async def respond(self, content: str, **kwargs):
        """
        针对该event，向当前会话发出响应
        """
        if not self._platform_manager:
            raise RuntimeError("PlatformManager 未注入 BaseEvent，无法 send()")
        return await self._platform_manager.send_message(self.platform, self.chat_stream_id, content, **kwargs)

    async def edit(self, message_id: str, content: str):
        if not self._platform_manager:
            raise RuntimeError("PlatformManager 未注入 BaseEvent，无法 edit()")
        return await self._platform_manager.edit_event(self.platform, message_id, content)

    async def delete(self, message_id: Optional[str] = None):
        if not self._platform_manager:
            raise RuntimeError("PlatformManager 未注入 BaseEvent，无法 delete()")
        msg_id = message_id or self.get("message_id")
        return await self._platform_manager.delete_event(self.platform, msg_id)

    # ----------------------------------------------------------------------
    # ------------------------ 事件派发（EventManager） ----------------------
    # ----------------------------------------------------------------------

    async def dispatch(self):
        """
        将事件重新送回 EventManager（扩展事件、插件触发事件等用途）
        """
        if not self._event_manager:
            raise RuntimeError("EventManager 未注入 BaseEvent，无法 dispatch()")
        return await self._event_manager.post(self)

    # ----------------------------------------------------------------------
    # ------------------------ 消息事件便捷方法（可选） -----------------------
    # ----------------------------------------------------------------------

    def get_plain_text(self) -> str:
        """
        如果是 MessageEvent，并且 event_data 含 message.plain_text 则返回。
        否则返回空字符串。
        """
        if self.event_data and hasattr(self.event_data, "plain_text"):
            return self.event_data.plain_text
        return ""

    def __repr__(self):
        return (
            f"\nevent类型: {self.event_type} event_id: {self.event_id} "
            f"\n平台: {self.platform} 用户: {self.user_info.user_nickname} 会话: {self.conversation_info.conversation_name}"
            f"\n内容: {self.event_data.__repr__()}"
        )


