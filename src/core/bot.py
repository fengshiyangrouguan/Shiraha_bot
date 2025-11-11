from typing import Optional
import logging


from src.core.chat import chat_manager
from src.core.replyer import Replyer
from src.plugin_system.manager import PluginManager
from src.plugin_system.plugin_planner import Planner

logger = logging.getLogger("main")

class MainSystem:
    """
    系统的“总指挥”，负责协调各个组件来处理消息。
    """
    def __init__(self):
        self.replyer = Replyer()
        self.plugin_manager = PluginManager()
        self.plugin_manager.load_plugins()  # 调用此方法来实际加载插件
        self.plugin_planner = Planner(self.plugin_manager)
        logger.info("MainSystem 已启动，并创建了 Replyer 实例。")

    async def handle_message(self, stream_id: str, user_id: str, message_content: str) -> Optional[str]:
        """
        处理单条传入消息的完整流程。

        Args:
            stream_id (str): 对话的唯一ID (例如 "private_12345" 或 "group_67890")。
            user_id (str): 发送消息用户的ID。
            message_content (str): 消息的文本内容。

        Returns:
            Optional[str]: 机器人生成的回复内容，如果无需回复或生成失败则为None。
        """
        logger.info(f"--- 收到新消息 ---\n来自: {stream_id}\n用户: {user_id}\n内容: {message_content}\n--------------------")

        # 1. 使用 chat_manager 获取或创建对应的聊天流 (ChatStream)
        chat_stream = chat_manager.get_or_create_stream(stream_id)
        logger.info(f"获取到 ChatStream: {chat_stream}")

        # 2. 将用户的新消息添加到聊天流中，以维护上下文
        chat_stream.add_message(user_id=user_id, content=message_content)
        logger.info(f"新消息已添加到 ChatStream，当前历史消息数: {len(chat_stream.messages)}")

        # 3. 调用 Planner 决定是否使用工具
        planner_results = await self.plugin_planner.plan_and_execute(message_content, chat_stream.messages)
        logger.info(f"Planner 返回的工具结果: {planner_results}")

        # 4. 把工具结果交给 Replyer
        reply_content, model_used = await self.replyer.generate_reply(
            chat_stream,
            message_content,
            tool_results=planner_results
        )

        if reply_content:
            logger.info(f"LLM ({model_used}) 已生成回复: {reply_content}")
            # 将机器人的回复也添加到聊天流中，以便下次对话时“记得”
            chat_stream.add_message(user_id="BOT", content=reply_content)
            logger.info("机器人回复已添加到 ChatStream。")
            return reply_content
        else:
            logger.warning("LLM 未能生成回复。")
            return None
