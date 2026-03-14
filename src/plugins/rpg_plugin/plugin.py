# src/plugins/rpg_plugin/plugin.py
from typing import List, Tuple, Type

from src.common.action_model.tool_result import ToolResult
from src.common.logger import get_logger
from src.plugin_system.base import BasePlugin, BaseTool, ToolInfo
from src.plugins.rpg_plugin.sub_agent.rpg_agent import RPGAgent
from src.cortices.qq_chat.data_model.qq_chat_data import QQChatData
from src.cortices.qq_chat.data_model.chat_stream import QQChatStream
from src.agent.world_model import WorldModel
from src.common.di.container import container

logger = get_logger("rpg_plugin_entry")


class RPGTool(BaseTool):
    """
    进入跑团模式的入口工具。
    """
    async def execute(self, conversation_id: str, reason: str, **kwargs) -> ToolResult:
        """
        执行逻辑：实例化子代理并启动其会话循环。
        这个方法将会是一个长阻塞操作，直到跑团会话结束。
        """
        logger.info(f"准备进入跑团模式 '{conversation_id}'.")
        
        try:
            self._world_model = container.resolve(WorldModel)
            qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
            chat_stream: QQChatStream = await qq_chat_data.get_or_create_stream_by_id(conversation_id)
            if not chat_stream:
                return ToolResult(
                    success=True,
                    summary="游戏规划运行正常，但没找到想开始游戏的会话",
                )
            # 1. 实例化跑团子代理
            agent = RPGAgent(chat_stream)
            
            # 2. 启动并等待会话结束
            # run_session 是一个 async aiohttp，它将持续运行直到满足退出条件
            final_result = await agent.run_session(reason=reason)
            
            # 3. 返回整个会话的最终结果
            return final_result
            
        except Exception as e:
            error_msg = f"启动或运行跑团子代理时发生严重错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolResult(
                success=False,
                summary="跑团插件遇到意外崩溃，已强制中止。",
                error_message=str(e)
            )


class RPGPlugin(BasePlugin):
    """
    跑团游戏插件。
    加载并提供 RPGTool 作为子代理的入口。
    """

    def get_plugin_tools(self) -> List[Tuple[ToolInfo, Type[BaseTool]]]:
        """
        获取插件提供的工具列表。
        """
        tool_info = self.get_declared_tool_info("start_an_TRPG_game")
        return [(tool_info, RPGTool)]
