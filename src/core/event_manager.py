import asyncio
import logging
from asyncio import Queue
from typing import Any, Awaitable, Callable, Dict, Type

# 设置日志
logger = logging.getLogger(__name__)

# --- 1. 标准化事件定义 ---
class BaseEvent:
    """
    事件基类，定义所有事件共有的属性。
    """
    def __init__(self, event_type: str, source: str, data: Any = None):
        self.event_type = event_type  # 事件类型，用于路由到对应的处理器
        self.source = source          # 事件来源，例如 'Telegram', 'Discord', 'Scheduler'
        self.data = data              # 事件携带的数据

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(type={self.event_type}, source={self.source}, data={self.data})>"

# --- 2. 事件管理器核心 ---
Handler = Callable[[BaseEvent], Awaitable[None]]

class EventManager:
    """
    事件管理器核心类。
    
    持有事件队列和事件处理器注册表。
    """
    def __init__(self):
        self._event_queue: Queue[BaseEvent] = Queue()
        self._event_handler: Handler | None = None
        self._running = False
        self._task: asyncio.Task | None = None

    def register_event_handler(self, handler: Handler):
        """
        注册一个事件处理器。
        
        Args:
            handler: 一个接收 BaseEvent 并返回 Awaitable 的异步函数。
        """
        if self._event_handler:
            logger.warning(f"事件处理器已存在，将被覆盖。")
        self._event_handler = handler
        logger.info(f"事件处理器已注册。")


    def unregister_event_handler(self):
        """
        取消注册事件处理器。
        """
        if self._event_handler:
            self._event_handler = None
            logger.info(f"事件处理器已取消注册。")
        else:
            logger.warning(f"尝试取消注册一个不存在的事件处理器。")


    async def post(self, event: BaseEvent):
        """
        将一个事件放入队列。
        这是供生产者（如平台适配器）调用的主要方法。
        """
        await self._event_queue.put(event)
    
    async def _consumer(self):
        """事件的消费者循环：处理耗时长、延迟高的任务。"""
        while self._running:
            try:
                event = await self._event_queue.get()
                logger.debug(f"消费者接收到事件: {event}")
                
                if self._event_handler:
                    try:
                        # 普通任务并发执行
                        asyncio.create_task(self._event_handler(event))
                    except Exception as e:
                        logger.error(f"执行事件处理器时出错: {e}", exc_info=True)
                else:
                    logger.warning(f"没有注册事件处理器。")
                
                self._event_queue.task_done()
            except asyncio.CancelledError:
                logger.info("事件消费者任务被取消。")
                break
            except Exception as e:
                logger.error(f"事件消费者循环出现意外错误: {e}", exc_info=True)


    def run(self):
        """
        启动事件管理器，同时启动两个消费者循环。
        """
        if not self._running:
            self._running = True
            
            # 启动消费者任务
            self._task = asyncio.create_task(self._consumer(), name="Consumer")
            
            logger.info("事件管理器已启动，消费者任务正在运行。")

    async def stop(self):
        """
        优雅地停止事件管理器。
        等待队列中的所有任务完成，然后停止消费者循环。
        """
        if not self._running:
            return
        logger.info("正在停止事件管理器...")

        # 等待队列中的所有项目被处理
        await self._event_queue.join()

        # 停止消费者循环
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # 预料之中的异常

        logger.info("事件管理器已停止。")


# --- 3. 全局单例 ---
# 在多数应用中，一个全局的事件管理器单例就足够了。
# 主程序在启动时初始化它，然后通过依赖注入或直接导入的方式提供给其他模块。
event_manager = EventManager()

# --- 示例用法 (可以放在 main.py 或其他启动脚本中) ---
async def example_usage():
    # 1. 定义一个事件处理器
    async def handle_message(event: BaseEvent):
        print(f"正在处理消息: 来自 {event.source} 的 {event.data}")
        await asyncio.sleep(1) # 模拟I/O操作

    # 2. 注册处理器
    event_manager.register_handler("new_message", handle_message)

    # 3. 启动事件管理器
    event_manager.run()

    # 4. 模拟生产者产生事件
    await event_manager.post(BaseEvent("new_message", "Telegram", {"text": "你好！"}))
    await event_manager.post(BaseEvent("new_message", "Discord", {"text": "Hello!"}))
    await event_manager.post(BaseEvent("other_event", "System", {})) # 这个事件将被忽略

    # 5. 停止事件管理器
    await event_manager.stop()

if __name__ == '__main__':

    
    # 运行示例
    asyncio.run(example_usage())
