import asyncio
import time
from typing import Optional

from src.common.logger import get_logger
from src.core.replyer import Replyer
from src.plugin_system.manager import PluginManager
from src.plugin_system.plugin_planner import Planner
from src.core.event_manager import EventManager
from src.platform.platform_manager import PlatformManager
from src.core.chat.event_processor import EventProcessor

logger = get_logger("main")

class MainSystem:
    """
    Shiraha_bot系统，负责协调各个组件来处理消息。
    """
    def __init__(self,global_config=None):
        # 依赖注入配置
        self.global_config = global_config

        self.event_manager: Optional[EventManager] = None
        self.platform_manager: Optional[PlatformManager] = None 
        
    async def initialize(self):
        """初始化主系统"""
        logger.info(f"正在唤醒{self.global_config.bot.nickname}......")
        # 初始化系统部件任务
        await asyncio.gather(self._init_components())

        logger.info(f"""
--------------------------------
全部系统初始化完成，{self.global_config.bot.nickname}已成功唤醒
--------------------------------
""")


    async def _init_components(self):
        """初始化系统组件"""
        init_start_time = time.time()

        # 添加在线时间统计任务
        # await async_task_manager.add_task(OnlineTimeRecordTask())

        # 添加统计信息输出任务
        # await async_task_manager.add_task(StatisticOutputTask())

        # 添加遥测心跳任务
        # await async_task_manager.add_task(TelemetryHeartBeatTask())

        # 启动API服务器
        # start_api_server()
        # logger.info("API服务器启动成功")

        self.event_manager = EventManager()
        self.event_processor = EventProcessor()
        self.platform_manager = PlatformManager(self.event_manager.post)
        # 将事件处理器注册到事件管理器
        self.event_manager.register_event_handler(self.event_processor.process_event)


        try:
            init_time = int(1000 * (time.time() - init_start_time))
            logger.info(f"初始化完成，神经元放电{init_time}次")
        except Exception as e:
            logger.error(f"启动大脑和外部世界失败: {e}")
            raise

    async def schedule_tasks(self):
        """调度定时任务"""
        while True:
            tasks = [
                self.app.run(),
                self.server.run(),
            ]

            await asyncio.gather(*tasks)



# 示例
async def main():
    """主函数"""
    system = MainSystem()
    await asyncio.gather(
        system.initialize(),
        system.schedule_tasks(),
    )


if __name__ == "__main__":
    asyncio.run(main())



