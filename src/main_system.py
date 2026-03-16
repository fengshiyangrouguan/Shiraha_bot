import asyncio
from pathlib import Path
from typing import Optional

from src.agent.agent_loop import AgentLoop
from src.agent.world_model import WorldModel
from src.common.config.config_service import ConfigService
from src.common.config.schemas.bot_config import BotConfig
from src.common.config.schemas.llm_api_config import LLMApiConfig
from src.common.database.database_manager import DatabaseManager
from src.common.di.container import container
from src.common.logger import get_logger
from src.common.tool_registry import ToolRegistry
from src.cortices.manager import CortexManager
from src.llm_api.factory import LLMRequestFactory
from src.memory_system.repositories.expression_pattern_repository import ExpressionPatternRepository
from src.memory_system.services.expression_learning_service import ExpressionLearningService
from src.memory_system.services.expression_selector_service import ExpressionSelectorService
from src.platform.platform_manager import PlatformManager
from src.plugin_system.core.plugin_loader import PluginLoader
from src.plugin_system.core.plugin_manager import PluginManager

logger = get_logger("main_system")


class MainSystem:
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service
        self.agent_loop: Optional[AgentLoop] = None
        self.world_model: Optional[WorldModel] = None
        self.cortex_manager: Optional[CortexManager] = None
        self.platform_manager: Optional[PlatformManager] = None
        self.plugin_loader: Optional[PluginLoader] = None
        self.plugin_manager: Optional[PluginManager] = None
        self.tool_registry: Optional[ToolRegistry] = None

    async def initialize(self):
        config_bot: BotConfig = self.config_service.get_config("bot")
        logger.info(f"开始初始化 {config_bot.persona.bot_name}...")

        await self._init_components()

        logger.info("--- 开始加载插件 ---")
        plugin_loader = container.resolve(PluginLoader)
        plugin_manager = container.resolve(PluginManager)
        plugin_infos = plugin_loader.load_plugins()
        plugin_manager.initialize_from_infos(plugin_infos)
        logger.info("--- 插件加载完成 ---")

        if self.cortex_manager:
            await self.cortex_manager.load_all_cortices()

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

        self.world_model = WorldModel()
        self.llm_request_factory = LLMRequestFactory()
        self.platform_manager = PlatformManager()
        self.database_manager = DatabaseManager()
        await self.database_manager.initialize_database()
        self.tool_registry = ToolRegistry()
        self.expression_pattern_repository = ExpressionPatternRepository(self.database_manager)
        self.expression_learning_service = ExpressionLearningService(
            self.database_manager,
            self.llm_request_factory,
        )
        self.expression_selector_service = ExpressionSelectorService(
            self.expression_pattern_repository,
            self.llm_request_factory,
            container.resolve(BotConfig),
        )

        self.plugin_loader = PluginLoader(plugin_root=Path("src/plugins"))
        self.plugin_manager = PluginManager(tool_registry=self.tool_registry)
        self.cortex_manager = CortexManager()

        container.register_instance(PluginLoader, self.plugin_loader)
        container.register_instance(PluginManager, self.plugin_manager)
        container.register_instance(ToolRegistry, self.tool_registry)
        container.register_instance(WorldModel, self.world_model)
        container.register_instance(LLMRequestFactory, self.llm_request_factory)
        container.register_instance(DatabaseManager, self.database_manager)
        container.register_instance(ExpressionPatternRepository, self.expression_pattern_repository)
        container.register_instance(ExpressionLearningService, self.expression_learning_service)
        container.register_instance(ExpressionSelectorService, self.expression_selector_service)
        container.register_instance(PlatformManager, self.platform_manager)
        container.register_instance(CortexManager, self.cortex_manager)

        self.agent_loop = AgentLoop()
        container.register_instance(AgentLoop, self.agent_loop)

        logger.info("--- 核心组件注册完成 ---")

    def _init_config(self):
        bot_config: BotConfig = self.config_service.get_config("bot")
        llm_api_config: LLMApiConfig = self.config_service.get_config("llm_api")

        container.register_instance(ConfigService, self.config_service)
        container.register_instance(LLMApiConfig, llm_api_config)
        container.register_instance(BotConfig, bot_config)

    async def schedule_tasks(self):
        while True:
            await asyncio.sleep(60)

    async def shutdown(self):
        logger.info("系统开始关闭...")
        if self.agent_loop:
            self.agent_loop.stop()

        if self.cortex_manager:
            await self.cortex_manager.shutdown_all_cortices()

        if self.platform_manager:
            await self.platform_manager.shutdown_all_adapters()

        await asyncio.sleep(1)
        logger.info("系统已关闭。")
