# src/plugins/hello_world_plugin/plugin.py

from typing import Dict, Any, List, Tuple, Type
from src.plugin_system.base import BasePlugin, BaseTool, ToolInfo
from src.common.di.container import container
from src.common.logger import get_logger
from src.cortices.qq_chat.api import QQChatAPI
# 导入 ToolResult
from src.common.action_model.tool_result import ToolResult

logger = get_logger("hello_world_plugin")

class SendHelloWorldTool(BaseTool):
    """
    一个简单的工具，用于发送 "Hello, World!"。
    """
    async def execute(self, conversation_id: str, conversation_type: str, **kwargs) -> ToolResult:
        """
        执行发送消息的逻辑，返回标准的 ToolResult。
        """
        logger.info(f"准备向 {conversation_id} ({conversation_type}) 发送 Hello World。")
        
        try:
            # 1. 从容器中解析 API Hub
            api = container.resolve(QQChatAPI)
            if not api:
                return ToolResult(
                    success=False, 
                    summary="发送失败：无法获取 QQChatAPI",
                    error_message="QQChatAPI is None. Ensure QQChatCortex is initialized."
                )
            
            # 2. 调用 API 发送消息
            # 注意：之前讨论过，如果是 lambda 注册，这里解析出来的是实例，直接调用方法即可
            await api.send_message(
                conversation_id=conversation_id,
                content="SAMURAIIIIIIIII!!!!!!!",
                conversation_type=conversation_type
            )
            
            summary = f"你在聊天大喊了一声“SAMURAI!!!!!!!!!”。"
            logger.info(summary)
            
            # 3. 返回标准的 ToolResult
            return ToolResult(
                success=True,
                summary=summary
            )
            
        except Exception as e:
            error_msg = f"发送 Hello World 消息时出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # 发生异常时返回失败状态
            return ToolResult(
                success=False,
                summary="发送 Hello World 任务异常中止",
                error_message=str(e)
            )

class HelloWorldPlugin(BasePlugin):
    """
    一个用于演示的简单插件。
    """
    def get_plugin_tools(self) -> List[Tuple[ToolInfo, Type[BaseTool]]]:
        """
        获取插件包含的工具列表
        """
        # 注意：这里需要确保你的插件基类支持 get_declared_tool_info 方法
        send_hello_world_info = self.get_declared_tool_info("send_hello_world")
        return [(send_hello_world_info, SendHelloWorldTool)]