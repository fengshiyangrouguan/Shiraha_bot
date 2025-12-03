# src/cortices/qq_chat/tools/send_quick_reply.py
from typing import Dict, Any

from src.cortices.tools_base import BaseTool
from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter
from src.agent.world_model import WorldModel
from src.cortices.qq_chat.chat.qq_chat_data import QQChatData


class ViewUnreadMsgTool(BaseTool):
    """
    向指定的QQ聊天对象（用户或群组）发送一条简单的、一次性的消息。
    适用于不需要深入对话的场景。
    """
    def __init__(self, world_model: WorldModel):
        self._world_model = world_model

    @property
    def scope(self) -> str:
        return "main"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "view_unread_message",
            "description": "查看所有未读消息",
            "parameters": {},
            "required_parameters": []
        }

    async def execute(self) -> str:
        """
        执行获取历史记录的逻辑。
        """
        try:
            # 1. 从 WorldModel 获取上下文信息
            qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
            history = qq_chat_data.get_all_streams_history_for_llm()
            return f"你了解到：{history}"
        except Exception as e:
            return f"获取历史记录时出错: {e}"
