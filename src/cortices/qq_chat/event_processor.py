import asyncio
import json
from typing import Any, Dict, Optional, List

from src.common.logger import get_logger
from src.agent.world_model import WorldModel
from src.common.event_model.event import Event
from src.system.di.container import container
from src.common.database.database_manager import DatabaseManager
from src.common.database.database_model import UserInfoDB, ConversationInfoDB, EventDB
from .qq_data import QQChatData
from .chat_stream import QQChatStream # 确保 QQChatStream 导入正确


logger = get_logger("qq_event_processor")

class QQChatEventProcessor:
    """
    QQ Chat Cortex 专用的事件处理器。
    负责从内部事件队列拉取事件，进行详细处理，
    更新 WorldModel 中的 QQChatData 状态，并将事件永久化存储到数据库。
    """
    def __init__(
        self,
        world_model: WorldModel,
        event_queue: asyncio.Queue[Event] # 接收 Cortex 传递过来的队列
    ):
        """
        初始化 QQ Chat 事件处理器。

        Args:
            world_model: WorldModel 的实例，用于与世界模型交互。
            event_queue: 用于接收事件的队列。
        """
        self.world_model:WorldModel = world_model
        self.database_manager = container.resolve(DatabaseManager)
        self._event_queue: asyncio.Queue[Event] = event_queue # 使用传入的队列
        logger.info("QQChatEventProcessor: 初始化完成。")

    async def run(self):
        """
        后台任务：从事件队列持续处理事件。
        """
        logger.info("QQChatEventProcessor: 事件处理循环已启动。")
        while True:
            event: Event = await self._event_queue.get()
            try:
                await self.process_event(event)
            except Exception as e:
                # 确保 conversation_info 存在再尝试获取 stream_id
                stream_id = f"{event.platform}_{event.conversation_info.conversation_id}" if event.conversation_info else "system_event"
                logger.error(f"处理事件失败 ({stream_id}): {e}", exc_info=True)
            finally:
                 self._event_queue.task_done()

    # post_event_to_queue 方法不再需要，因为 _event_queue 已在 __init__ 中传入，Cortex 直接调用 _event_queue.put
    async def post_event_to_queue(self, event: Event) -> Any:
        await self._event_queue.put(event)
        return

    async def process_event(self, event: Event):
        """
        处理单个事件。
        根据事件类型执行不同的处理逻辑，并更新 WorldModel 和数据库。

        Args:
            event (Event): 待处理的事件对象。
        """
        
        # 1. 从 WorldModel 获取/创建 QQChatData 顶层状态对象
        qq_data: QQChatData = await self.world_model.get_data("qq_chat_data", QQChatData)
        if not qq_data:
            qq_data = QQChatData()

        # 2. 获取/创建对应的 QQChatStream
        # 对于没有 conversation_info 的系统事件，可以将其视为一个特殊的流
        stream_id = f"{event.platform}_{event.conversation_info.conversation_id}" if event.conversation_info else f"{event.platform}_system_event"
        chat_stream = qq_data.get_or_create_stream(stream_id)

        # 3. 预处理事件数据，生成 LLM 纯文本 (如果适用)
        if event.event_type == "message":

            await event.event_data.process_to_context()
        
        # 4. 更新 QQChatStream 的内部状态 (滑动窗口、未读计数等)
        chat_stream.add_event(event)
        
        # 5. 将更新后的 QQChatData 完整保存回 WorldModel
        await self.world_model.save_data("qq_chat_data", qq_data)
        logger.debug(f"QQChatData for stream ({stream_id}) 已更新并保存到 WorldModel。")

        # 6. 数据库永久化存储 (EventDB, UserInfoDB, ConversationInfoDB)
        # 注意：这里需要将 Pydantic/dataclass 对象转换为 SQLModel 对象
        
        user_db: Optional[UserInfoDB] = None
        if event.user_info:
            user_db = UserInfoDB(
                user_id=event.user_info.user_id,
                user_nickname=event.user_info.user_nickname,
                user_cardname=event.user_info.user_cardname
            )
            await self.database_manager.upsert(user_db)
            print(json.dumps(user_db.model_dump(), ensure_ascii=False, indent=4))
            logger.debug(f"UserInfoDB (ID: {user_db.user_id}) 已保存。")

        conversation_db: Optional[ConversationInfoDB] = None
        if event.conversation_info:
            conversation_db = ConversationInfoDB(
                conversation_id=event.conversation_info.conversation_id,
                conversation_type=event.conversation_info.conversation_type,
                conversation_name=event.conversation_info.conversation_name,
                parent_id=event.conversation_info.parent_id,
                platform_meta=event.conversation_info.platform_meta
            )
            await self.database_manager.upsert(conversation_db)
            print(json.dumps(conversation_db.model_dump(), ensure_ascii=False, indent=4))

            logger.debug(f"ConversationInfoDB (ID: {conversation_db.conversation_id}) 已保存。")
        
        event_content = event.event_data.LLM_plain_text
        event_tags_list = list(event.tags) if event.tags else []

        event_db = EventDB(
            event_id=event.event_id,
            platform=event.platform,
            event_type=event.event_type,
            time=event.time,
            conversation_id=event.conversation_info.conversation_id if event.conversation_info else None,
            conversation_type=event.conversation_info.conversation_type if event.conversation_info else None,
            conversation_name=event.conversation_info.conversation_name if event.conversation_info else None,
            user_id=event.user_info.user_id if event.user_info else None,
            user_nickname=event.user_info.user_nickname if event.user_info else None,
            user_cardname=event.user_info.user_cardname if event.user_info else None,
            tags=event_tags_list,
            event_content=event_content, # 存储为 JSON 字符串
            event_metadata=event.event_data.metadata
        )
        await self.database_manager.upsert(event_db)
        print(json.dumps(event_db.model_dump(), ensure_ascii=False, indent=4))
        logger.info(f"EventDB (ID: {event_db.event_id}) 已永久化存储。")

        logger.info(f"事件 ({event.event_type}) 针对流 ({stream_id}) 处理完成。")

