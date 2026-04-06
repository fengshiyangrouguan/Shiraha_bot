import asyncio
from typing import Dict, List, Callable, Any, Awaitable
from src.common.logger import get_logger

logger = get_logger("ipc_bus")

class MessageBus:
    """
    内部消息总线 (Signal Bus)
    用于传递 Task 信号 (如: FINISH, CALL_HELP, ERROR)
    """
    def __init__(self):
        # 存储订阅者 { "signal_type": [callback_funcs] }
        self._subscribers: Dict[str, List[Callable[[Any], Awaitable[None]]]] = {}

    def subscribe(self, signal_type: str, callback: Callable[[Any], Awaitable[None]]):
        """订阅特定类型的信号"""
        if signal_type not in self._subscribers:
            self._subscribers[signal_type] = []
        self._subscribers[signal_type].append(callback)
        logger.debug(f"Subscribed to signal: {signal_type}")

    async def publish(self, signal_type: str, payload: Any):
        """发布信号，异步触发所有回调"""
        if signal_type not in self._subscribers:
            return

        logger.info(f"📡 Signal Published: {signal_type} | Payload: {str(payload)[:50]}...")
        
        # 并发触发所有订阅了该信号的处理器 (如 InterruptHandler 或 Scheduler)
        tasks = [callback(payload) for callback in self._subscribers[signal_type]]
        if tasks:
            await asyncio.gather(*tasks)

# 全局总线实例通常由 DI 容器管理