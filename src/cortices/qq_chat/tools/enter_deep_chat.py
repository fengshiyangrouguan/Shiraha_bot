# src/cortices/qq_chat/tools/enter_deep_chat.py
from typing import Dict, Any, TYPE_CHECKING, List
import json
from src.common.action_model.tool_result import ToolResult
from src.cortex_system.tools_base import BaseTool
from src.common.di.container import container
from src.cortices.qq_chat.data_model.qq_chat_data import QQChatData
from src.common.database.database_manager import DatabaseManager
from src.cortices.qq_chat.chat.deep_chat_sub_agent import DeepChatSubAgent
from src.agent.world_model import WorldModel
from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter
from src.cortices.qq_chat.cortex import QQChatCortex
from src.llm_api.factory import LLMRequestFactory
from src.cortices.qq_chat.chat.replyer import QQReplyer
from src.common.logger import get_logger

logger = get_logger("qq_deep_chat")

if TYPE_CHECKING:
    
    from src.cortex_system.manager import CortexManager

class EnterDeepChatTool(BaseTool):
    """
    一个“启动器”工具，用于启动一个专注于单个会话的“深度聊天子智能体”。
    调用此工具将会阻塞主循环，直到子智能体完成其任务并退出。
    """
    def __init__(self, adapter: QQNapcatAdapter, cortex:"QQChatCortex",cortex_manager: "CortexManager"):
        super().__init__(cortex_manager)
        self.adapter = adapter
        self.cortex = cortex
        self._world_model = container.resolve(WorldModel)
        self.llm_request_factory = container.resolve(LLMRequestFactory)
        self.database_manager = container.resolve(DatabaseManager)

        self.replyer = QQReplyer(world_model=self._world_model,adapter=self.adapter,llm_request_factory=self.llm_request_factory,database_manager=self.database_manager,cortex=self.cortex)
        self.deep_chat_agent = DeepChatSubAgent(adapter=self.adapter,cortex=self.cortex,replyer=self.replyer)

    @property
    def scope(self) -> List[str]:
        return ["qq_app","quick_reply"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "enter_deep_chat",
            "description": "当你想更深度长时间参与/观察某聊天/对话，或主动开启话题时，使用此工具。",
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
        chat_stream = await qq_chat_data.get_or_create_stream_by_id(conversation_id)
        logger.info(f"准备进入深度聊天 -> {conversation_id}")
        if not chat_stream:
            return ToolResult(success=False, summary=f"找不到会话 {conversation_id}，无法进入深度聊天。", error_message=f"Conversation {conversation_id} not found.")

        # 2. 调用并等待子智能体运行结束
        # 整个主循环将在这里“暂停”，直到 deep_chat_sub_agent 完成它的所有内部循环并返回结果
        available_tools = self.cortex_manager.get_tool_schemas(scopes=["deep_chat"])
        available_tools_str = json.dumps(available_tools, ensure_ascii=False, indent=2)
        final_result = await self.deep_chat_agent.run(
            intent=reason,
            chat_stream=chat_stream,
            available_tools = available_tools_str
        )
        chat_stream.mark_as_read()  # 标记为已读，结束后续的未读提醒
        # 3. 将子智能体的最终总结作为此工具的结果返回
        return final_result
