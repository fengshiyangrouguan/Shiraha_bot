# main.py
import asyncio

# 导入核心服务和模块
from src.common.config.config_service import ConfigService
from src.common.config.schemas.llm_api_config import LLMApiConfig
from src.common.config.schemas.bot_config import BotConfig
from src.system.main_system import MainSystem

from src.common.logger import setup_logger
logger = setup_logger("Application")

async def main():
    """
    应用的主异步入口函数。
    """
    system = None
    try:
        # --- 1. 预加载并校验所有核心配置 ---
        logger.info("正在加载和校验核心配置文件...")
        
        # 通过调用 get_config，我们强制服务去加载、解析和验证文件。
        # 如果文件不存在或格式错误，程序会在这里立即失败，而不会启动主系统。
        config_service = ConfigService()
        bot_config:BotConfig = config_service.get_config("bot", BotConfig)
        llm_api_config:LLMApiConfig = config_service.get_config("llm_api", LLMApiConfig)
        
        logger.info(f"配置校验成功！机器人名称: {bot_config.bot_name}")
        logger.info(f"共加载了 {len(llm_api_config.api_providers)} 个API供应商和 {len(llm_api_config.models)} 个模型。")

        # --- 2. 初始化并启动主系统 ---
        logger.info("正在初始化主系统...")
        system = MainSystem(config_service=config_service)
        
        await system.initialize()
        
        # 启动 Agent 的核心循环
        system.agent_loop.start()
        
        # 运行主系统的其他永久性任务（如果有）
        await system.schedule_tasks()

    except (RuntimeError, ValueError) as e:
        logger.error(f"应用启动失败: {e}")
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("收到关闭信号...")
    finally:
        if system:
            logger.info("正在执行关机程序...")
            await system.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("程序被强制退出。")
