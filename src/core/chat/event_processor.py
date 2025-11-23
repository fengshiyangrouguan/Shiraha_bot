import logging
from typing import Optional, List, Callable, Awaitable, Dict, Any
from dataclasses import dataclass, field

from src.common.event_model.event import Event
from src.core.chat import ChatManager
from src.core.replyer import Replyer
from src.plugin_system.manager import PluginManager
from src.plugin_system.plugin_planner import Planner
from src.plugin_system.event_types import EventType  # 导入新的事件类型
from src.common.event_model.event_data import Message

logger = logging.getLogger("message_processor")

# 定义消息处理器特有的事件处理函数类型
MessageEventHandler = Callable[[Event], Awaitable[bool]]  # 返回 True 继续处理，False 中断

class EventProcessor:
    """
    负责处理 Shiraha_bot 接收到的消息。
    模拟 MaiBot 消息处理流程，支持插件注册事件处理器。
    """
    def __init__(self, chat_manager: ChatManager, plugin_manager: PluginManager, planner: Planner, replyer: Replyer):
        self.chat_manager = chat_manager
        self.plugin_manager = plugin_manager
        self.planner = planner
        self.replyer = replyer
        self._event_handlers: Dict[EventType, List[MessageEventHandler]] = {}

    def register_event_handler(self, event_type: EventType, handler: MessageEventHandler):
        """注册消息处理器内部的事件处理器"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        logger.debug(f"已注册事件类型 {event_type.value} 的处理器")

    async def _dispatch_event(self, event_type: EventType, event: Event) -> bool:
        """
        分发内部事件。如果任何处理器返回 False 或事件被标记为停止传播，则中断处理。
        返回 True 表示继续处理，False 表示中断。
        """
        for handler in self._event_handlers.get(event_type, []):
            try:
                result = await handler(event)
                if not result or event.is_stopped():  # 改动1：兼容 stop_propagation
                    return False
            except Exception as e:
                logger.error(
                    f"事件类型 {event_type.value} 的事件处理器执行出错: {e}", exc_info=True
                )
        return True

    async def process_event(self, event: Event) -> Optional[str]:
        """
        执行事件处理管道。
        """

        user_id = getattr(event.user_info, "user_id", "UNKNOWN")
        logger.debug(
            f"开始处理事件 {getattr(event, 'event_id', 'UNKNOWN')}, "
            f"聊天流ID: {event.chat_stream_id}, 用户: {user_id}")

        try:
            # 1. 消息段处理
            if event.event_type == 'message':
                message: Optional[Message] = event.event_data
                await message.process_segments()  # 改动3：加 hasattr 检查，防止非消息事件报错

            # 2. ON_MESSAGE_PRE_PROCESS 事件
            if not await self._dispatch_event(EventType.ON_MESSAGE_PRE_PROCESS, event):
                logger.info(f"ON_MESSAGE_PRE_PROCESS 事件拦截了事件 {getattr(event, 'event_id', 'UNKNOWN')}")
                return None

            # 3. 获取或创建聊天流，并添加消息
            chat_stream = self.chat_manager.get_or_create_stream(event.chat_stream_id)
            content = getattr(event, "processed_plain_text", "")
            chat_stream.add_event(user_id=user_id, content=content)  # user_id 可能为 None

            # 4. ON_MESSAGE 事件
            if not await self._dispatch_event(EventType.ON_MESSAGE, event):
                logger.info(f"ON_MESSAGE 事件拦截了事件 {getattr(event, 'event_id', 'UNKNOWN')}")
                return None

            # 5. 规划器决定工具使用/响应
            planner_results = await self.planner.plan_and_execute(
                getattr(event, "processed_plain_text", ""),
                chat_stream.messages,
                event
            )
            if hasattr(event, "llm_response_tool_call"):
                event.llm_response_tool_call = planner_results

            # 6. POST_LLM 事件
            if not await self._dispatch_event(EventType.POST_LLM, event):
                logger.info(f"POST_LLM 事件拦截了事件 {getattr(event, 'event_id', 'UNKNOWN')}")
                return None

            # 7. LLM 回复生成
            reply_content, model_used = await self.replyer.generate_reply(
                chat_stream,
                getattr(event, "processed_plain_text", ""),
                tool_results=planner_results
            )
            if hasattr(event, "llm_response_content"):
                event.llm_response_content = reply_content
                event.llm_response_model = model_used

            # 8. AFTER_LLM 事件
            if not await self._dispatch_event(EventType.AFTER_LLM, event):
                logger.info(f"AFTER_LLM 事件拦截了事件 {getattr(event, 'event_id', 'UNKNOWN')}")
                return None

            # 9. 使用最终回复
            final_reply = getattr(event, "llm_response_content", reply_content)

            if final_reply:
                chat_stream.add_message(user_id="BOT", content=final_reply)
                logger.info(f"机器人回复: '{final_reply}'")
                return final_reply
            else:
                logger.warning("LLM 未生成回复或被事件处理器清空。")
                return None
            
        except Exception as e:
            logger.error(f"处理命令时出错: {e}")
            return False, None, True  # 出错时继续处理消息
