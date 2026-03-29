# src/cortices/qq_chat/cortex.py
import asyncio
from typing import Any, Dict, Optional, List

from src.common.logger import get_logger
from src.agent.world_model import WorldModel
from src.cortices.base_cortex import BaseCortex
from src.cortices.tools_base import BaseTool
from src.common.event_model.event import Event
from .config.config_schema import CortexConfigSchema
from src.cortices.qq_chat.data_model.qq_chat_data import QQChatData
from .chat.event_processor import QQChatEventProcessor
from src.cortices.manager import CortexManager
from .chat.sticker_system.sticker_manager import StickerManager
from src.extensions.social_damper import SocialDamper
from .api import QQChatAPI

from src.common.di.container import container
from src.platform.platform_manager import PlatformManager
from src.platform.platform_base import BasePlatformAdapter
from src.llm_api.factory import LLMRequestFactory
from src.common.database.database_manager import DatabaseManager


logger = get_logger("qq_chat")

class QQChatCortex(BaseCortex):
    """
    QQ 聊天皮层，负责管理与 QQ 平台的所有交互。
    该类协调 Adapter 和 EventProcessor，将平台事件转化为对 WorldModel 的更新，
    并向 CortexManager 提供其专属的工具。
    """
    def __init__(self):
        super().__init__()
        self.adapter: Optional[BasePlatformAdapter] = None
        self.event_processor: Optional[QQChatEventProcessor] = None
        self._process_events_task: Optional[asyncio.Task] = None
        self.llm_request_factory: Optional[LLMRequestFactory] = None
        self.database_manager: Optional[DatabaseManager] = None
        self.sticker_manager: Optional[StickerManager] = None
        self.social_damper: Optional[SocialDamper] = None

    @property
    def cortex_name(self) -> str:
        return "qq_chat"

    async def setup(self, world_model: WorldModel, config: CortexConfigSchema, cortex_manager: CortexManager):
        """
        启动 QQ 聊天皮层。
        此方法设置并启动所有必需的组件。
        """
        await super().setup(world_model, config, cortex_manager)
        logger.info(f"正在启动...")

        self.llm_request_factory = container.resolve(LLMRequestFactory)
        self.database_manager = container.resolve(DatabaseManager)
        self.event_processor = QQChatEventProcessor(world_model,config.bot_id)   
        platform_manager: PlatformManager = container.resolve(PlatformManager)

        self.sticker_manager = StickerManager(self.database_manager)
        await self.sticker_manager.start()
        self.social_damper = SocialDamper(self.llm_request_factory,self._world_model)

        container.register_factory(StickerManager, lambda: self.sticker_manager)
        container.register_factory(SocialDamper, lambda: self.social_damper)
        
        adapter_config = self.config.adapter

        try:
            self.adapter = await platform_manager.register_and_start(
                adapter_config, self.event_processor.post_event_to_queue
            )
            logger.info(f"已启动并注册适配器: '{self.adapter.adapter_id}'")

        except Exception as e:
            logger.error(f"启动适配器 '{adapter_config.adapter_id}' 失败: {e}")

        # 3. 启动事件处理器
        self._process_events_task = asyncio.create_task(self.event_processor.run())

        # 4. 注册 API Hub
        if self.adapter:
            api_hub = QQChatAPI(self.adapter, cortex_manager)
            container.register_factory(QQChatAPI, lambda: api_hub)
            logger.info("QQChatAPI 已成功注册到容器。")

        logger.info(f"启动完成。")

    async def post_event_to_processor(self, event: Event):
        """将一个事件（通常是出站事件）发送到事件处理器队列中进行持久化和状态更新。"""
        await self.event_processor.post_event_to_queue(event)

    async def get_cortex_summary(self):
        qq_chat_data:QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
        if qq_chat_data is None:
            return "QQ 尚未初始化。"
        summary = await qq_chat_data.get_global_perception_report()
        return summary

    async def teardown(self):
        """
        关闭 QQ 聊天皮层。
        安全地停止适配器和事件处理任务。
        """
        logger.info(f"正在关闭...")
        
        # 1. 停止事件处理任务
        if self._process_events_task:
            self._process_events_task.cancel()
            try:
                await self._process_events_task
            except asyncio.CancelledError:
                logger.info(f"事件处理任务已取消。")
        
        # 2. 停止所有适配器
        platform_manager: PlatformManager = container.resolve(PlatformManager)
        try:
            if self.adapter: # 适配器可能未成功启动
                await platform_manager.shutdown_adapter(self.adapter.adapter_id)
                del self.adapter
                logger.info(f"适配器 '{self.adapter.adapter_id}' 已停止。")
        except Exception as e:
            if self.adapter:
                logger.error(f"停止适配器 '{self.adapter.adapter_id}' 失败: {e}")
            else:
                logger.error(f"停止适配器失败，适配器未实例化: {e}")
    
        logger.info(f"已关闭。")
