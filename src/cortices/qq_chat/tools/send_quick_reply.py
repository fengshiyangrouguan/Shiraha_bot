# src/cortices/qq_chat/tools/send_quick_reply.py
from typing import Dict, Any

from src.cortices.tools_base import BaseTool
from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter
from src.agent.world_model import WorldModel
from src.cortices.qq_chat.chat.qq_chat_data import QQChatData


class SendQuickReplyTool(BaseTool):
    """
    向指定的QQ聊天对象（用户或群组）发送一条简单的、一次性的消息。
    适用于不需要深入对话的场景。
    """
    def __init__(self, world_model: WorldModel,adapter:QQNapcatAdapter):
        self._world_model = world_model
        self.adapter = adapter

    @property
    def scope(self) -> str:
        return "main"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "send_quick_qq_reply",
            "description": "向指定的QQ聊天对象（用户或群组）发送一条简单的消息。适用于不想或不需要深入聊天的场景。简单回应一下，回复的平淡一些，简短一些，不要描述动作，尽量少使用标点，一条回复可分几次发送，",
            "parameters": {
                "conversation_id": {
                    "type": "string",
                    "description": "目标聊天（用户或群组）的ID。"
                },
                "content": {
                    "type": "string",
                    "description": "要发送的消息文本内容。"
                }
            },
            "required_parameters": ["conversation_id", "content"]
        }

    async def execute(self, conversation_id: str, content: str) -> str:
        """
        执行发送快速回复的逻辑。
        """
        try:
            # 1. 从 WorldModel 获取上下文信息
            qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
            if not qq_chat_data or conversation_id not in qq_chat_data.streams:
                return f"错误：在WorldModel中未找到ID为 {conversation_id} 的聊天记录，无法确定发送平台。"

            chat_stream = qq_chat_data.streams[conversation_id]
            conversation_info = chat_stream.conversation_info
            
            adapter_id = self.adapter.adapter_id

            await self.adapter.message_api.send_text(conversation_info, content)
            return f"消息 '{content}' 已成功发送至 {conversation_id}。"

        except Exception as e:
            return f"发送消息时出错: {e}"
