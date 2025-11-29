import asyncio
import time
import logging
from typing import Optional

# 导入 DI 容器
from .di.container import container

# 导入核心服务和模块
from src.common.config.config_service import ConfigService
from src.common.config.schemas.bot_config import BotConfig
from src.common.config.schemas.llm_api_config import LLMApiConfig
from src.common.logger import get_logger
from src.common.database.database_manager import DatabaseManager
from src.llm_api.factory import LLMRequestFactory
from src.agent.agent_loop import AgentLoop
from src.features.manager import FeatureManager

logger = get_logger("main_system")

class MainSystem:
    """
    Shiraha_bot主系统，负责组装协调各个组件。
    """
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service

    async def initialize(self):
        """初始化主系统"""
        config_bot: BotConfig = self.config_service.get_config("bot")
        logger.info(f"正在唤醒 {config_bot.persona.bot_name}......")
        
        await asyncio.gather(self._init_components())

        logger.info(f"""
--------------------------------
全部系统初始化完成，{config_bot.persona.bot_name} 已成功唤醒
--------------------------------
""")

    async def _init_components(self):
        """初始化系统组件"""
        logger.info("--- 正在设置应用依赖 ---")

        self.llm_request_factory = LLMRequestFactory()
        self.agent_loop = AgentLoop()
        self.feature_manager = FeatureManager()
        
        bot_config:BotConfig = self.config_service.get_config("bot")
        llm_api_config:LLMApiConfig = self.config_service.get_config("llm_api")

        container.register_instance(ConfigService, self.config_service)
        container.register_instance(LLMApiConfig, llm_api_config)
        container.register_instance(BotConfig, bot_config)

        container.register_instance(LLMRequestFactory, self.llm_request_factory)
        container.register_instance(FeatureManager, self.feature_manager)

        logger.info("--- 所有依赖已注册 ---")



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
        # 等待后台任务完成清理
        await asyncio.sleep(1)
        logger.info("系统已关闭。")




