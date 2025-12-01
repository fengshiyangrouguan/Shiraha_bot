import asyncio
import time
import logging
from typing import Optional, List

# 导入 DI 容器
from .di.container import container

# 导入核心服务和模块
from src.common.logger import get_logger
from src.common.config.config_service import ConfigService
from src.common.config.schemas.bot_config import BotConfig
from src.common.config.schemas.llm_api_config import LLMApiConfig
from src.common.database.database_manager import DatabaseManager
from src.llm_api.factory import LLMRequestFactory
from src.agent.agent_loop import AgentLoop
from src.cortices.manager import CortexManager
from src.platform.platform_manager import PlatformManager # 导入 PlatformManager

from src.agent.world_model import WorldModel # Import WorldModel

logger = get_logger("main_system")

class MainSystem:
    """
    Shiraha_bot主系统，负责组装协调各个组件。
    """
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service
        self.agent_loop: Optional[AgentLoop] = None # Initialize as Optional
        self.world_model: Optional[WorldModel] = None
        self.cortex_manager: Optional[CortexManager] = None
        self.platform_manager: Optional[PlatformManager] = None # 新增：PlatformManager 实例
        # self.impetus_descriptions: List[str] = [] # 移除：MotiveEngine 直接接收

    async def initialize(self):
        """初始化主系统"""
        config_bot: BotConfig = self.config_service.get_config("bot")
        logger.info(f"正在唤醒 {config_bot.persona.bot_name}......")
        
        await self._init_components()

        # Load all cortices after core components are initialized
        if self.cortex_manager:
            await self.cortex_manager.load_all_cortices()

        logger.info(f"""
--------------------------------
全部系统初始化完成，{config_bot.persona.bot_name} 已成功唤醒
--------------------------------
""")

    async def _init_components(self):
        """初始化系统组件"""
        logger.info("--- 正在设置应用依赖 ---")

        self._init_config()
        
        
        self.world_model = WorldModel()
        self.llm_request_factory = LLMRequestFactory()
        self.platform_manager = PlatformManager() # 实例化 PlatformManager
        self.database_manager = DatabaseManager()
        await self.database_manager.initialize_database()  # 初始化数据库

        self.cortex_manager = CortexManager()
        
        container.register_instance(WorldModel, self.world_model)
        container.register_instance(LLMRequestFactory, self.llm_request_factory)
        container.register_instance(DatabaseManager, self.database_manager)
        container.register_instance(PlatformManager, self.platform_manager) # 注册 PlatformManager
        container.register_instance(CortexManager, self.cortex_manager)
        

        self.agent_loop = AgentLoop() # AgentLoop 的实例化将由用户自行协调其参数传递
        container.register_instance(AgentLoop, self.agent_loop)

        logger.info("--- 所有依赖已注册 ---")


    def _init_config(self):
        """初始化并注册配置服务和各类配置"""
        bot_config:BotConfig = self.config_service.get_config("bot")
        llm_api_config:LLMApiConfig = self.config_service.get_config("llm_api")

        container.register_instance(ConfigService, self.config_service)
        container.register_instance(LLMApiConfig, llm_api_config)
        container.register_instance(BotConfig, bot_config)


    async def schedule_tasks(self):
        """(占位符) 调度未来的定时任务"""
        while True:
            # logger.debug("主调度器正在运行...")
            await asyncio.sleep(60)

    async def shutdown(self):
        """关闭所有服务"""
        logger.info("系统正在关闭...")
        if self.agent_loop:
            self.agent_loop.stop()
        
        if self.cortex_manager:
            await self.cortex_manager.shutdown_all_cortices()

        if self.platform_manager: # 新增：关闭所有适配器
            await self.platform_manager.shutdown_all_adapters()

        # 等待后台任务完成清理
        await asyncio.sleep(1)
        logger.info("系统已关闭。")




