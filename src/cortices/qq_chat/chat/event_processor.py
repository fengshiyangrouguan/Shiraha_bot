import asyncio
import json
from typing import Any, Dict, Optional, List

from src.common.logger import get_logger
from src.agent.world_model import WorldModel
from src.common.event_model.event import Event
from .qq_chat_data import QQChatData
from .chat_stream import QQChatStream # 确保 QQChatStream 导入正确


logger = get_logger("QQEventProcessor")

class QQChatEventProcessor:
    """
    QQ Chat Cortex 专用的事件处理器。
    负责从内部事件队列拉取事件，进行详细处理，
    更新 WorldModel 中的 QQChatData 状态，并将事件永久化存储到数据库。
    """
    def __init__(
        self,
        world_model: WorldModel,
        bot_id: str
    ):
        """
        初始化 QQ Chat 事件处理器。

        Args:
            world_model: WorldModel 的实例，用于与世界模型交互。
            event_queue: 用于接收事件的队列。
        """
        self.world_model:WorldModel = world_model
        self.qq_chat_data: Optional[QQChatData] = None  
        self.event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self.bot_id = bot_id

        logger.info("QQChatEventProcessor: 初始化完成。")

    async def run(self):
        """
        后台任务：初始化 QQChatData，并持续处理事件。
        """
        self.qq_chat_data = await self.world_model.get_cortex_data("qq_chat_data")
        if self.qq_chat_data is None:
            self.qq_chat_data = QQChatData(bot_id=self.bot_id)
            await self.world_model.save_cortex_data("qq_chat_data", self.qq_chat_data)

        logger.info("QQChatEventProcessor: 事件处理循环已启动。")


        while True:
            event: Event = await self.event_queue.get()
            try:
                await self.process_event(event)
            except Exception as e:
                # 确保 conversation_info 存在再尝试获取 stream_id
                stream_id = f"{event.conversation_info.conversation_id}" if event.conversation_info else "system_event"
                logger.error(f"处理事件失败 ({stream_id}): {e}", exc_info=True)
            finally:
                 self.event_queue.task_done()

    async def post_event_to_queue(self, event: Event) -> Any:
        await self.event_queue.put(event)
        return

    async def process_event(self, event: Event):
        """
        处理单个事件。
        根据事件类型执行不同的处理逻辑，并更新 WorldModel 和数据库。

        Args:
            event (Event): 待处理的事件对象。
        """
        
        # 2. 获取/创建对应的 QQChatStream
        # 对于没有 conversation_info 的系统事件，可以将其视为一个特殊的流
        chat_stream:QQChatStream = self.qq_chat_data.get_or_create_stream(event.conversation_info)
        

        # 3. 预处理事件数据，生成 LLM 纯文本
        if event.event_type == "message":
            await event.event_data.process_to_context()
        
        # 4. 更新 QQChatStream 的内部状态 (滑动窗口、未读计数等)
        await chat_stream.add_event(event)

        # 5. 将更新后的 QQChatData 和 上下文完整保存回 WorldModel
        self.world_model.notifications["QQ聊天"] = self.qq_chat_data.total_unread_count
        await self.world_model.save_cortex_data("qq_chat_data", self.qq_chat_data)

        logger.debug(f"QQChatData for stream ({chat_stream.stream_id}) 已更新并保存到 WorldModel。")

        # 6. 数据库永久化存储 (EventDB, UserInfoDB, ConversationInfoDB)
        # 注意：这里需要将 Pydantic/dataclass 对象转换为 SQLModel 对象
