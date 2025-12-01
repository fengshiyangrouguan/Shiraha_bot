import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict
import logging

from pydantic import BaseModel # 导入 BaseModel
from src.common.event_model.event import Event
# 平台配置Schema不再作为基类的一部分，而是作为具体适配器的泛型配置


# 定义 post 方法的类型签名，它是一个接收 Event 并返回协程的函数
PostMethod = Callable[[Event], Coroutine[Any, Any, None]]

logger = logging.getLogger(__name__) # 初始化日志记录器

class BasePlatformAdapter(ABC): # 类名修改为 BasePlatformAdapter
    """
    平台适配器的抽象基类。
    所有特定平台的适配器都应继承自此类，并实现其定义的所有抽象方法。
    """

    def __init__(self, adapter_id: str, platform_type: str, config: BaseModel, post_method: PostMethod):
        """
        初始化适配器基类。

        Args:
            adapter_id (str): 此适配器实例的唯一标识符。
            platform_type (str): 此适配器所服务的平台类型（例如 'qq_napcat', 'discord'）。
            config (BaseModel): 此适配器的配置对象，已由 Pydantic Schema 验证。
                                  它应至少包含 adapter_id 和 platform_type 字段。
            post_method (PostMethod): 一个异步函数，用于将事件提交给主事件管理器。
        """
        self.platform_type = platform_type # 平台类型
        self.config = config            # 适配器的配置对象 (由具体Adapter的Schema定义)
        self.adapter_id = adapter_id    # 适配器实例ID
        self.post_method = post_method  # 事件提交函数
        self._is_running = False        # 适配器运行状态标志

    def commit_event(self, event: Event):
        """
        将一个事件提交到主事件队列中。
        这是一个非阻塞方法，它会创建一个任务来异步执行 `post_method`。

        Args:
            event (Event): 要提交的事件对象。
        """
        # 使用 asyncio.create_task 来调用异步的 post_method，避免阻塞当前适配器的执行流程。
        asyncio.create_task(self.post_method(event))
        logger.debug(f"适配器 '{self.adapter_id}' 已提交事件: {event.event_type}")

    @abstractmethod
    def run(self) -> asyncio.Task:
        """
        启动适配器。
        所有继承此基类的具体适配器都必须实现此方法。
        这个方法应该包含启动和运行适配器所需的所有逻辑。
        它必须返回一个 `asyncio.Task` 对象，以便于 `PlatformManager` 进行监控和管理。
        """
        pass

    @abstractmethod
    async def terminate(self):
        """
        优雅地停止适配器。
        所有继承此基类的具体适配器都必须实现此方法。
        这个方法应该包含清理和关闭适配器所需的所有逻辑。
        """
        pass