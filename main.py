# main.py
import asyncio

from src.common.config.config_service import ConfigService
from src.common.config.schemas.bot_config import BotConfig
from src.common.config.schemas.llm_api_config import LLMApiConfig
from src.common.logger import get_logger
from src.main_system import MainSystem

logger = get_logger("main")


async def main():
    """
    应用主入口。

    新版启动流程不再依赖旧的 AgentLoop，而是直接交给 MainSystem
    去完成事件驱动主链的初始化和启动。
    """
    system = None
    wait_forever = asyncio.Event()

    try:
        logger.info("正在加载和校验核心配置文件...")

        config_service = ConfigService()
        bot_config: BotConfig = config_service.get_config("bot")
        llm_api_config: LLMApiConfig = config_service.get_config("llm_api")

        logger.info(f"配置校验成功！机器人名称: {bot_config.persona.bot_name}")
        logger.info(f"共加载了 {len(llm_api_config.api_providers)} 个 API 供应商和 {len(llm_api_config.models)} 个模型。")

        logger.info("正在初始化主系统...")
        system = MainSystem(config_service=config_service)
        await system.initialize()

        # 保持主进程常驻，让 EventLoop 和各个后台任务持续运行。
        await wait_forever.wait()

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
