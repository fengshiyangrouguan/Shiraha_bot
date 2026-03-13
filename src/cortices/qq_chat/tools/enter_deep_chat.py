# src/cortices/qq_chat/tools/enter_deep_chat.py
from typing import Dict, Any

from src.common.action_model.tool_result import ToolResult
from src.cortices.tools_base import BaseTool
from src.common.di.container import container
from src.cortices.qq_chat.data_model.qq_chat_data import QQChatData
from src.common.database.database_manager import DatabaseManager
from src.cortices.qq_chat.chat.deep_chat_sub_agent import DeepChatSubAgent
from src.agent.world_model import WorldModel

class EnterDeepChatTool(BaseTool):
    """
    一个“启动器”工具，用于启动一个专注于单个会话的“深度聊天子智能体”。
    调用此工具将会阻塞主循环，直到子智能体完成其任务并退出。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 该工具需要获取 WorldModel 来访问数据
        self._world_model: WorldModel = container.resolve(WorldModel)
        self.database_manager: DatabaseManager = container.resolve(DatabaseManager)
        # 创建子智能体实例
        self.deep_chat_sub_agent = DeepChatSubAgent()

    @property
    def scope(self) -> str:
        return "batch_plan"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "enter_deep_chat",
            "description": "当你认为一个会话非常重要或有趣，想更深度长时间参与/观察某聊天，或主动开启话题时，使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "聊天ID。"
                    },
                    "reason": {
                        "type": "string",
                        "description": "你为什么想对这个会话进行深度参与。深度聊天的原因/意图"
                    }
                },
                "required": ["conversation_id", "reason"]
            }
        }

    async def execute(self, conversation_id: str, reason: str, **kwargs) -> ToolResult:
        """
        执行方法：获取会话流，并启动子智能体。
        """
        # 1. 获取会话上下文
        qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
        chat_stream = qq_chat_data.get_or_create_stream_by_id(conversation_id, self.database_manager)
        
        if not chat_stream:
            return ToolResult(success=False, summary=f"找不到会话 {conversation_id}，无法进入深度聊天。", error_message=f"Conversation {conversation_id} not found.")

        # 2. 调用并等待子智能体运行结束
        # 整个主循环将在这里“暂停”，直到 deep_chat_sub_agent 完成它的所有内部循环并返回结果
        final_result = await self.deep_chat_sub_agent.run(
            initial_intent=reason,
            chat_stream=chat_stream
        )

        # 3. 将子智能体的最终总结作为此工具的结果返回
        return final_result
