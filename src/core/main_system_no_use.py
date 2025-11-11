import asyncio
import time


logger = get_logger("main")


class MainSystem:
    def __init__(self):
        # 使用消息API替代直接的FastAPI实例
        pass
    async def initialize(self):
        """初始化系统组件"""
        logger.info(f"正在唤醒{global_config.bot.nickname}......")

        await asyncio.gather(self._init_components())

        logger.info(f"""
--------------------------------
全部系统初始化完成，{global_config.bot.nickname}已成功唤醒
--------------------------------
""")

    async def _init_components(self):
        """初始化组件"""
        init_start_time = time.time()

        # 添加在线时间统计任务
        # 添加统计信息输出任务
        # 添加遥测心跳任务
        # 启动API服务器
        # start_api_server()
        # logger.info("API服务器启动成功")
        # 加载所有actions，包括默认的和插件的
        # 启动情绪管理器
        # 初始化聊天管理器

        logger.info("聊天管理器初始化成功")

        # 将message_process消息处理函数注册到api.py的消息处理基类中
        # 触发 ON_START 事件
        # logger.info("已触发 ON_START 事件")
        try:
            init_time = int(1000 * (time.time() - init_start_time))
            logger.info(f"初始化完成，神经元放电{init_time}次")
        except Exception as e:
            logger.error(f"启动大脑和外部世界失败: {e}")
            raise

    async def schedule_tasks(self):
        """调度定时任务"""


async def main():
    """主函数"""
    system = MainSystem()
    await asyncio.gather(
        system.initialize(),
        system.schedule_tasks(),
    )


if __name__ == "__main__":
    asyncio.run(main())