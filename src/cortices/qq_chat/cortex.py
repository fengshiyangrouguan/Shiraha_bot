# src/cortices/qq_chat/cortex.py
import asyncio
from typing import Any, Dict, Optional

from src.agent.world_model import WorldModel
from src.cortices.base_cortex import BaseCortex
from src.common.event_model.event import Event
from .config.config_schema import CortexConfigSchema
from .event_processor import QQChatEventProcessor

from src.system.di.container import container
from src.platform.platform_manager import PlatformManager
from src.platform.platform_base import BasePlatformAdapter


class QQChatCortex(BaseCortex):
    """
    QQ 聊天皮层，负责管理与 QQ 平台的所有交互。
    该类协调 Adapter 和 EventProcessor，将平台事件转化为对 WorldModel 的更新。
    """
    def __init__(self):
        self.config: Optional[CortexConfigSchema] = None
        self.adapters: Dict[str, BasePlatformAdapter] = {}
        self.event_processor: Optional[QQChatEventProcessor] = None

        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._process_events_task: Optional[asyncio.Task] = None

    async def setup(self, world_model: WorldModel, config: CortexConfigSchema):
        """
        启动 QQ 聊天皮层。
        此方法设置并启动所有必需的组件：
        1. 平台适配器 (Platform Adapter)，用于接收外部事件。
        2. 事件处理器 (Event Processor)，用于在后台处理事件。

        Args:
            world_model: WorldModel 的实例。
            config: 此 Cortex 的配置。
        """
        print(f"正在启动...")
        self.config = config

        self.event_processor = QQChatEventProcessor(world_model, self._event_queue)
        platform_manager: PlatformManager = container.resolve(PlatformManager)

        # 1. 设置并启动平台适配器
        adapter_config = self.config.adapter
        post_event_to_queue = self.event_processor.post_event_to_queue
        try:
            adapter_instance = await platform_manager.register_and_start(
                adapter_config, post_event_to_queue
            )
            self.adapters[adapter_instance.adapter_id] = adapter_instance
            print(f"已启动并注册适配器: '{adapter_instance.adapter_id}'")
        except Exception as e:
            print(f"启动适配器 '{adapter_config.adapter_id}' 失败: {e}")

        # 2. 启动事件处理器
        self._process_events_task = asyncio.create_task(self.event_processor.run())

        print(f"启动完成。")

    async def teardown(self):
        """
        关闭 QQ 聊天皮层。
        安全地停止适配器和事件处理任务。
        """
        print(f"正在关闭...")
        
        # 1. 停止事件处理任务
        if self._process_events_task:
            self._process_events_task.cancel()
            try:
                await self._process_events_task
            except asyncio.CancelledError:
                print(f"事件处理任务已取消。")
        
        # 2. 停止所有适配器
        platform_manager: PlatformManager = container.resolve(PlatformManager)
        for adapter_id in list(self.adapters.keys()):
            try:
                await platform_manager.shutdown_adapter(adapter_id)
                del self.adapters[adapter_id]
                print(f"适配器 '{adapter_id}' 已停止。")
            except Exception as e:
                print(f"停止适配器 '{adapter_id}' 失败: {e}")
        
        print(f"已关闭。")
