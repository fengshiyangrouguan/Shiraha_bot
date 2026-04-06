"""
Interrupt Handler - 中断处理器

接收来自 Cortex 的异步信号，进行优先级筛选，转发给 EventLoop
"""
import asyncio
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel

from src.common.logger import get_logger
from src.common.di.container import container
from src.core.task.models import Priority

logger = get_logger("interrupt_handler")


class CortexSignal(BaseModel):
    """来自 Cortex 的信号"""
    signal_type: str           # 信号类型
    source_cortex: str         # 来源 cortex
    source_target: str         # 来源目标（用户ID、群ID等）
    content: str               # 信号内容
    priority: str = "medium"   # 优先级
    timestamp: float = time.time()    # 时间戳
    metadata: Dict[str, Any] = {}     # 元数据

    def _parse_priority(self) -> Priority:
        """将字符串优先级转换为 Priority 枚举"""
        priority_map = {
            "critical": Priority.CRITICAL,
            "high": Priority.HIGH,
            "medium": Priority.MEDIUM,
            "low": Priority.LOW,
        }
        return priority_map.get(self.priority.lower(), Priority.MEDIUM)


class InterruptHandler:
    """
    内核中断处理器 (Interrupt Handler)

    负责接收来自 Cortex 的异步信号，进行优先级筛选，转发给 EventLoop
    """

    def __init__(self):
        # 中断屏蔽字 (Interrupt Mask)
        self.interrupt_mask_level: Priority = Priority.LOW

        # EventLoop 引用（延迟绑定）
        self._event_loop = None

        # 信号统计
        self._stats = {
            "received": 0,
            "masked": 0,
            "queued": 0
        }

    def bind_event_loop(self, event_loop) -> None:
        """
        在 MainSystem 完成 EventLoop 实例化后进行显式绑定。

        这样可以避免 InterruptHandler 在 import/初始化阶段就去反向依赖 EventLoop，
        从而打断事件驱动主链的装配顺序。
        """
        self._event_loop = event_loop

    def _priority_rank(self, priority: Priority) -> int:
        """获取优先级权重"""
        if priority == Priority.CRITICAL:
            return 3
        if priority == Priority.HIGH:
            return 2
        if priority == Priority.MEDIUM:
            return 1
        return 0

    async def handle_cortex_signal(self, signal: CortexSignal):
        """
        处理来自 Cortex 的信号

        Args:
            signal: CortexSignal 对象
        """
        self._stats["received"] += 1

        logger.info(f"📥 收到 Cortex 信号: {signal.signal_type} from {signal.source_cortex} -> {signal.source_target}")

        # 转换优先级
        priority = signal._parse_priority()

        # 中断过滤
        if self._priority_rank(priority) < self._priority_rank(self.interrupt_mask_level):
            self._stats["masked"] += 1
            logger.debug(f"信号被屏蔽: {priority.value} < {self.interrupt_mask_level.value}")
            return

        # 提交到 EventLoop
        await self._submit_to_event_loop(signal, priority)

    async def handle_external_event(
        self,
        source_cortex: str,
        signal_type: str,
        content: str,
        source_target: str = "",
        priority: str = "medium",
        **metadata
    ):
        """
        简化的外部事件入口（兼容旧接口）

        Args:
            source_cortex: 来源 cortex 名称
            signal_type: 信号类型
            content: 信号内容
            source_target: 来源目标
            priority: 优先级
            **metadata: 额外元数据
        """
        signal = CortexSignal(
            signal_type=signal_type,
            source_cortex=source_cortex,
            source_target=source_target,
            content=content,
            priority=priority,
            metadata=metadata
        )

        await self.handle_cortex_signal(signal)

    async def _submit_to_event_loop(self, signal: CortexSignal, priority: Priority):
        """提交信号到 EventLoop"""
        try:
            # 确保 EventLoop 已初始化
            if self._event_loop is None:
                try:
                    from src.core.kernel.event_loop import EventLoop
                    self._event_loop = container.resolve(EventLoop)
                except Exception:
                    logger.warning("EventLoop 不可用，无法提交信号")
                    return

            # 创建中断信号
            from src.core.kernel.event_loop import InterruptSignal
            interrupt_signal = InterruptSignal(
                source_cortex=signal.source_cortex,
                target_id=signal.source_target or f"{signal.source_cortex}_unknown",
                content=signal.content,
                priority=priority,
                raw_data=signal.metadata,
                timestamp=signal.timestamp
            )

            # 提交到队列
            await self._event_loop.submit_interrupt(interrupt_signal)
            self._stats["queued"] += 1

            logger.info(f"✅ 信号已排队: {signal.signal_type} -> {signal.source_target}")

        except Exception as e:
            logger.error(f"提交信号到 EventLoop 失败: {e}", exc_info=True)

    def set_mask_level(self, level: Priority):
        """
        设置中断屏蔽等级

        Args:
            level: 屏蔽等级，低于此等级的信号会被忽略
        """
        self.interrupt_mask_level = level
        logger.info(f"中断屏蔽等级设置为: {level.value}")

    def reset_mask(self):
        """重置中断屏蔽等级"""
        self.interrupt_mask_level = Priority.LOW
        logger.info(f"中断屏蔽等级已重置: {self.interrupt_mask_level.value}")

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self._stats.copy()

    def is_masked(self, signal: CortexSignal) -> bool:
        """
        检查信号是否会被屏蔽

        Args:
            signal: CortexSignal 对象

        Returns:
            True 如果会被屏蔽
        """
        priority = signal._parse_priority()
        return self._priority_rank(priority) < self._priority_rank(self.interrupt_mask_level)
