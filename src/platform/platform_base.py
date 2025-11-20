import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict

# 导入事件基类
from ..core.event_manager import BaseEvent

# 定义 post 方法的类型签名，它是一个接收 BaseEvent 并返回协程的函数
PostMethod = Callable[[BaseEvent], Coroutine[Any, Any, None]]

class PlatformAdapterBase(ABC):
    """
    平台适配器的抽象基类。
    所有特定平台的适配器都应继承自此类，并实现其定义的所有抽象方法。
    """

    def __init__(self, post_method: PostMethod, platform_config: Dict[str, Any]):
        """
        初始化适配器基类。

        :param post_method: 一个异步函数，用于将事件提交给 EventManager。
        :param platform_config: 该平台适配器的特定配置。
        """
        self.post_method = post_method
        self.config = platform_config
        self._is_running = False

    def commit_event(self, event: BaseEvent):
        """
        将一个事件提交到主事件队列中。
        这是一个非阻塞方法，它会创建一个任务来异步执行 post_method。

        :param event: 要提交的事件对象，必须是 BaseEvent 的实例。
        """
        # 使用 asyncio.create_task 来调用异步的 post_method，避免阻塞当前执行流程
        asyncio.create_task(self.post_method(event))

    @abstractmethod
    def run(self) -> asyncio.Task:
        """
        启动适配器。
        这个方法应该包含启动和运行适配器所需的所有逻辑，
        例如：启动一个 WebSocket 服务器、连接到一个远程服务等。
        它应该返回一个 asyncio.Task 对象，以便于管理者进行监控。
        """
        pass

    @abstractmethod
    async def terminate(self):
        """
        优雅地停止适配器。
        这个方法应该包含清理和关闭适配器所需的所有逻辑，
        例如：关闭服务器、断开连接、释放资源等。
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取适配器的元数据。
        元数据是一个字典，包含关于适配器的信息，例如名称、描述、ID等。

        :return: 一个包含元数据的字典。
        """
        pass