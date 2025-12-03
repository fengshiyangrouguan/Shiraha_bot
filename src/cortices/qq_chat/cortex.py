# src/cortices/qq_chat/cortex.py
import asyncio
from typing import Any, Dict, Optional, List

from src.agent.world_model import WorldModel
from src.cortices.base_cortex import BaseCortex
from src.cortices.tools_base import BaseTool
from src.common.event_model.event import Event
from .config.config_schema import CortexConfigSchema
from .chat.event_processor import QQChatEventProcessor
from src.cortices.manager import CortexManager

from src.system.di.container import container
from src.platform.platform_manager import PlatformManager
from src.platform.platform_base import BasePlatformAdapter

# 导入工具类
from .tools.enter_chat_mode import EnterChatModeTool
from .tools.send_quick_reply import SendQuickReplyTool
from .tools.view_unread_msg import ViewUnreadMsgTool


class QQChatCortex(BaseCortex):
    """
    QQ 聊天皮层，负责管理与 QQ 平台的所有交互。
    该类协调 Adapter 和 EventProcessor，将平台事件转化为对 WorldModel 的更新，
    并向 CortexManager 提供其专属的工具。
    """
    def __init__(self):
        self.config: Optional[CortexConfigSchema] = None
        self.adapter: Optional[BasePlatformAdapter] = None
        self.event_processor: Optional[QQChatEventProcessor] = None
        self._world_model: Optional[WorldModel] = None
        self._cortex_manager: Optional[CortexManager] = None

        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._process_events_task: Optional[asyncio.Task] = None


    async def setup(self, world_model: WorldModel, config: CortexConfigSchema, cortex_manager: CortexManager):
        """
        启动 QQ 聊天皮层。
        此方法设置并启动所有必需的组件。

        Args:
            world_model: WorldModel 的实例。
            config: 此 Cortex 的配置。
            cortex_manager: CortexManager 的实例。
        """
        print(f"QQChatCortex: 正在启动...")
        self.config = config
        self._world_model = world_model
        self._cortex_manager = cortex_manager

        self.event_processor = QQChatEventProcessor(world_model, self._event_queue)
        platform_manager: PlatformManager = container.resolve(PlatformManager)

        # 1. 设置并启动平台适配器
        adapter_config = self.config.adapter

        try:
            self.adapter = await platform_manager.register_and_start(
                adapter_config, self.event_processor.post_event_to_queue
            )
            print(f"QQChatCortex: 已启动并注册适配器: '{self.adapter.adapter_id}'")

        except Exception as e:
            print(f"QQChatCortex: 启动适配器 '{adapter_config.adapter_id}' 失败: {e}")

        # 2. 启动事件处理器
        self._process_events_task = asyncio.create_task(self.event_processor.run())

        print(f"QQChatCortex: 启动完成。")

    def get_tools(self) -> List[BaseTool]:
        """
        实例化并返回此 Cortex 提供的所有工具。
        """
        if not self._world_model or not self._cortex_manager:
            raise RuntimeError("QQChatCortex尚未完全初始化，无法获取工具。")
        
        return [
            # EnterChatModeTool(
            #     world_model=self._world_model,
            #     cortex_manager=self._cortex_manager, 
            #     adapter_id=self.config.adapter.adapter_id),
            SendQuickReplyTool(self._world_model,self.adapter),
            ViewUnreadMsgTool(self._world_model)
        ]
        
    async def teardown(self):
        """
        关闭 QQ 聊天皮层。
        安全地停止适配器和事件处理任务。
        """
        print(f"QQChatCortex: 正在关闭...")
        
        # 1. 停止事件处理任务
        if self._process_events_task:
            self._process_events_task.cancel()
            try:
                await self._process_events_task
            except asyncio.CancelledError:
                print(f"QQChatCortex: 事件处理任务已取消。")
        
        # 2. 停止所有适配器
        platform_manager: PlatformManager = container.resolve(PlatformManager)
        try:
            await platform_manager.shutdown_adapter(self.adapter.adapter_id)
            del self.adapter
            print(f"QQChatCortex: 适配器 '{self.adapter.adapter_id}' 已停止。")
        except Exception as e:
            print(f"QQChatCortex: 停止适配器 '{self.adapter.adapter_id}' 失败: {e}")
    
        print(f"QQChatCortex: 已关闭。")
