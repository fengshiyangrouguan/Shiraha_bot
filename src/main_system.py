import asyncio
from pathlib import Path
from typing import Optional

from src.agent.world_model import WorldModel
from src.common.config.config_service import ConfigService
from src.common.config.schemas.bot_config import BotConfig
from src.common.config.schemas.llm_api_config import LLMApiConfig
from src.common.database.database_manager import DatabaseManager
from src.common.di.container import container
from src.common.logger import get_logger
from src.common.tool_registry import ToolRegistry
from src.core.kernel import EventLoop, InterruptHandler, KernelInterpreter, Scheduler
from src.core.memory import LongTermMemory, MemoryRetriever, UnifiedMemory
from src.core.task.task_manager import TaskManager
from src.core.task.task_store import TaskStore
from src.cortex_system import CortexManager
from src.llm_api.factory import LLMRequestFactory
from src.platform.platform_manager import PlatformManager
from src.plugin_system.core.plugin_loader import PluginLoader
from src.plugin_system.core.plugin_manager import PluginManager

logger = get_logger("main_system")


class MainSystem:
    """
    主系统装配器。

    这一版的职责非常明确：
    1. 注册事件驱动主链所需的全部依赖。
    2. 加载插件与 Cortex。
    3. 启动 EventLoop，作为系统唯一主循环。
    """

    def __init__(self, config_service: ConfigService):
        self.config_service = config_service

        self.world_model: Optional[WorldModel] = None
        self.cortex_manager: Optional[CortexManager] = None
        self.platform_manager: Optional[PlatformManager] = None
        self.plugin_loader: Optional[PluginLoader] = None
        self.plugin_manager: Optional[PluginManager] = None
        self.tool_registry: Optional[ToolRegistry] = None
        self.database_manager: Optional[DatabaseManager] = None

        self.long_term_memory: Optional[LongTermMemory] = None
        self.memory_retriever: Optional[MemoryRetriever] = None
        self.unified_memory: Optional[UnifiedMemory] = None

        self.task_store: Optional[TaskStore] = None
        self.task_manager: Optional[TaskManager] = None
        self.scheduler: Optional[Scheduler] = None
        self.interrupt_handler: Optional[InterruptHandler] = None
        self.kernel_interpreter: Optional[KernelInterpreter] = None
        self.event_loop: Optional[EventLoop] = None

    async def initialize(self):
        """
        初始化并启动事件驱动主链。

        这里会一次性完成：
        - 基础服务注册
        - 内核组件装配
        - 插件加载
        - Cortex 加载
        - EventLoop 启动
        """
        config_bot: BotConfig = self.config_service.get_config("bot")
        logger.info(f"开始初始化 {config_bot.persona.bot_name}...")

        await self._init_components()
        await self._load_plugins()
        await self._load_cortices()
        await self._start_event_loop()

        logger.info(
            f"""
--------------------------------
系统初始化完成，{config_bot.persona.bot_name} 已准备就绪
--------------------------------
"""
        )

    async def _init_components(self):
        logger.info("--- 初始化核心组件 ---")

        self._init_config()

        # 先初始化最基础、无容器依赖的服务。
        self.database_manager = DatabaseManager()
        await self.database_manager.initialize_database()
        self.tool_registry = ToolRegistry()
        self.platform_manager = PlatformManager()

        self.plugin_loader = PluginLoader(plugin_root=Path("src/plugins"))
        self.plugin_manager = PluginManager(tool_registry=self.tool_registry)

        # 记忆系统要先于 WorldModel 注册，这样 WorldModel 初始化时就能直接拿到统一记忆。
        self.long_term_memory = LongTermMemory()
        self.memory_retriever = MemoryRetriever(self.long_term_memory)
        self.unified_memory = UnifiedMemory()
        await self.unified_memory.initialize(
            long_term_memory=self.long_term_memory,
            retriever=self.memory_retriever,
        )

        container.register_instance(DatabaseManager, self.database_manager)
        container.register_instance(ToolRegistry, self.tool_registry)
        container.register_instance(PlatformManager, self.platform_manager)
        container.register_instance(PluginLoader, self.plugin_loader)
        container.register_instance(PluginManager, self.plugin_manager)
        container.register_instance(LongTermMemory, self.long_term_memory)
        container.register_instance(MemoryRetriever, self.memory_retriever)
        container.register_instance(UnifiedMemory, self.unified_memory)

        # LLM 工厂会被 Cortex 和 Planner 共同依赖，因此需要尽早注册。
        llm_request_factory = LLMRequestFactory()
        container.register_instance(LLMRequestFactory, llm_request_factory)

        # WorldModel 依赖 BotConfig 与 UnifiedMemory，因此必须放在上面两个之后。
        self.world_model = WorldModel()
        await self.world_model.initialize_memory(self.unified_memory)
        container.register_instance(WorldModel, self.world_model)

        # 下面开始注册事件驱动内核的全部组件。
        self.task_store = TaskStore()
        container.register_instance(TaskStore, self.task_store)

        self.task_manager = TaskManager(self.task_store)
        container.register_instance(TaskManager, self.task_manager)

        self.scheduler = Scheduler()
        container.register_instance(Scheduler, self.scheduler)

        self.interrupt_handler = InterruptHandler()
        container.register_instance(InterruptHandler, self.interrupt_handler)

        # KernelInterpreter 与 MainPlanner 在初始化阶段都会解析 CortexManager，
        # 因此必须先注册，再创建这些依赖它的对象。
        self.cortex_manager = CortexManager()
        container.register_instance(CortexManager, self.cortex_manager)

        self.kernel_interpreter = KernelInterpreter()
        container.register_instance(KernelInterpreter, self.kernel_interpreter)

        self.event_loop = EventLoop()
        container.register_instance(EventLoop, self.event_loop)

        # EventLoop 完成注册后，再把引用反向注入给中断处理器，避免初始化阶段循环依赖。
        self.interrupt_handler.bind_event_loop(self.event_loop)

        logger.info("--- 核心组件注册完成 ---")

    async def _load_plugins(self):
        logger.info("--- 开始加载插件 ---")
        plugin_infos = self.plugin_loader.load_plugins()
        self.plugin_manager.initialize_from_infos(plugin_infos)
        logger.info("--- 插件加载完成 ---")

    async def _load_cortices(self):
        if not self.cortex_manager:
            return

        await self.cortex_manager.load_all_cortices(
            signal_callback=self._handle_cortex_signal,
            skill_manager=None,
        )

    async def _start_event_loop(self):
        if self.event_loop:
            await self.event_loop.start()

    async def _handle_cortex_signal(self, signal):
        """
        Cortex 信号统一入口。

        BaseCortex 发来的信号会先进入这里，再被转换成内核可识别的中断信号。
        这样 MainSystem 可以把“感知层”与“内核层”的边界固定下来。
        """
        if not self.interrupt_handler:
            logger.warning("InterruptHandler 尚未初始化，信号被忽略")
            return

        await self.interrupt_handler.handle_external_event(
            source_cortex=signal.source_cortex,
            signal_type=signal.signal_type,
            content=signal.content,
            source_target=signal.source_target,
            priority=signal.priority,
            **(signal.metadata or {}),
        )

    def _init_config(self):
        bot_config: BotConfig = self.config_service.get_config("bot")
        llm_api_config: LLMApiConfig = self.config_service.get_config("llm_api")

        container.register_instance(ConfigService, self.config_service)
        container.register_instance(LLMApiConfig, llm_api_config)
        container.register_instance(BotConfig, bot_config)

    async def shutdown(self):
        logger.info("系统开始关闭...")

        if self.event_loop:
            await self.event_loop.stop()

        if self.cortex_manager:
            await self.cortex_manager.shutdown_all_cortices()

        if self.platform_manager:
            await self.platform_manager.shutdown_all_adapters()

        if self.unified_memory and hasattr(self.unified_memory, "shutdown"):
            await self.unified_memory.shutdown()

        if self.database_manager:
            await self.database_manager.close()

        await asyncio.sleep(0.2)
        logger.info("系统已关闭。")

    async def submit_debug_request(self, content: str):
        """
        向事件驱动主链提交一条调试台请求。
        """
        if not self.event_loop:
            raise RuntimeError("EventLoop 尚未初始化")
        await self.event_loop.submit_debug_input(content)
