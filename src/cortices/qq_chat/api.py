# src/cortices/qq_chat/api.py
from typing import TYPE_CHECKING
from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter
from src.common.event_model.info_data import ConversationInfo

if TYPE_CHECKING:
    from src.cortices.manager import CortexManager

class QQChatAPI:
    """
    专为插件系统提供的、稳定的QQ聊天相关API。
    """
    def __init__(self, adapter: QQNapcatAdapter, cortex_manager: "CortexManager"):
        self.adapter = adapter
        self.cortex_manager = cortex_manager

    async def send_message(self, conversation_id: str, content: str, conversation_type: str = 'group'):
        """
        发送一条消息到指定的会话。

        Args:
            conversation_id: 会话ID (群号或用户QQ号)。
            content: 消息内容。
            conversation_type: 会话类型 ('group' 或 'private')。
        """
        if not self.adapter:
            raise RuntimeError("QQChatAPI's adapter is not initialized.")

        # 构建一个临时的 ConversationInfo 对象用于发送
        convo_info = ConversationInfo(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            conversation_name="Unknown" # 发送时通常不需要名字
        )
        
        await self.adapter.message_api.send_text(convo_info, content)
