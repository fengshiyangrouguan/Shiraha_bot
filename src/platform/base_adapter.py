import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Awaitable, Callable, Optional
from asyncio import Queue, Event, CancelledError, TimeoutError, wait_for

from src.logger import logger
# 假设 ResponsePool 也是一个通用依赖
from src.response_pool import ResponsePool 
from src.constants import MSG_TYPE

# 定义核心消息处理函数的类型契约
CoreHandlerType = Callable[[Any], Awaitable[None]]
SecurityManagerType = Any # 假设 SecurityManager 类
MessageMapperType = Any # 假设 MessageMapper 类
EventBusHandlerType = Callable[[Dict[str, Any]], Awaitable[None]] # 新增事件处理入口

class BaseAdapter(ABC):
    """
    Adapter 基类：定义了所有适配器的核心 I/O 流程、并发原语和生命周期管理。
    所有子类必须实现抽象方法来提供平台特有的 I/O 逻辑。
    """

    def __init__(
        self, 
        config: Any, 
        core_system_handler: CoreHandlerType, 
        response_pool: ResponsePool,
        security_manager: SecurityManagerType,
        message_mapper: MessageMapperType,
        event_bus_handler: EventBusHandlerType, # 新增：系统事件处理依赖
    ):
        # 核心依赖 (通过 DI 注入)
        self.config = config
        self.core_system_handler = core_system_handler
        self.response_pool = response_pool
        self.security_manager = security_manager
        self.message_mapper = message_mapper
        self.event_bus_handler = event_bus_handler # 事件总线入口
        
        # 内部并发组件
        self.message_queue: Queue = Queue()
        self.should_stop: Event = Event()
        self.consumer_task: Optional[asyncio.Task] = None
        
        # 抽象连接对象（由子类在运行时填充）
        self.platform_connection: Any = None 
        # 可以在子类中重命名为 self.server_connection, self.client_socket 等

    # --- 抽象方法：必须由子类实现 (平台特有逻辑) ---

    @abstractmethod
    async def _start_platform_io(self) -> None:
        """
        I/O 生产者。实现平台特有的网络接收逻辑（如 ws.serve, TCP/UDP 监听等）。
        它必须将接收到的原始消息推入 self.message_queue。
        """
        raise NotImplementedError

    @abstractmethod
    async def _stop_io_connection(self) -> None:
        """
        停止 I/O 生产者。实现关闭平台特有连接和服务器的逻辑。
        eg:关闭 ws.WebSocketServer 或断开客户端连接。
        """
        raise NotImplementedError

 
    # --- 核心方法：基类实现 (平台通用逻辑) ---

    async def _process_message_queue(self):
        """
        三路消息路由器：从队列取出消息，通过 Mapper 分类，路由到正确的处理流程。
        """
        while not self.should_stop.is_set():
            try:
                raw_message = await wait_for(self.message_queue.get(), timeout=0.5) 
                
        
                # Mapper 负责将平台数据分类，基类只负责路由
                # 如果子类严格分流，Mapper 永远不会返回 API_RESPONSE
                message_type = self.message_mapper.get_message_type(raw_message)
                
                if message_type == MSG_TYPE.APP_MESSAGE:
                    # 路径 A: 应用消息 (聊天/指令) -> 过滤 -> 转换 -> 核心业务
                    if await self.security_manager.check_allow_to_chat(raw_message):
                        message_base = await self.message_mapper.handle_raw_message(
                            raw_message, 
                            self.platform_connection 
                        )
                        await self.core_system_handler(message_base)

                elif message_type == MSG_TYPE.SYSTEM_EVENT:
                    # 路径 B: 系统事件 (心跳/状态更新) -> 事件总线
                    await self.event_bus_handler(raw_message)
                    
                else:
                    # 路径 C: 剩下的一切 (理论上不会到这里)
                    logger.warning(f"Adapter: 队列中出现未知的消息类型 ({message_type})，已丢弃。")
                    
                self.message_queue.task_done()
                
            except TimeoutError:
                continue
            except CancelledError:
                break
            except Exception as e:
                logger.exception(f"消费者任务处理异常: {e}")
        
        self.should_stop.set()
    async def run(self):
        """Adapter 主入口：启动消费者和 I/O 生产者。"""
        logger.info(f"Adapter: 正在启动 {self.__class__.__name__}...")
        self.consumer_task = asyncio.create_task(self._process_message_queue(), name=f"{self.__class__.__name__}Consumer")
        try:
            await self._start_platform_io()
        except CancelledError:
            pass
        except Exception as e:
            logger.error(f"Adapter I/O 生产者异常退出: {e}")
            
        if self.consumer_task and not self.consumer_task.done():
            await self.consumer_task
        
        logger.info(f"{self.__class__.__name__} 退出完成。")


    def stop(self):
        """优雅退出接口。"""
        logger.warning(f"{self.__class__.__name__} 收到停止信号。")
        self.should_stop.set()
        
        asyncio.create_task(self._stop_io_connection())
        
        if self.consumer_task and not self.consumer_task.done():
            self.consumer_task.cancel()

        logger.info(f"{self.__class__.__name__} 停止流程已启动。")