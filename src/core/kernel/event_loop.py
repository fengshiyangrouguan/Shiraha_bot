"""
Event Loop - 事件驱动核心循环

替代旧的固定循环，采用事件驱动模式：
1. Cortex 信号 → InterruptHandler 接收
2. InterruptHandler → 触发内核重调度
3. Scheduler 选择焦点任务执行
4. 只有无任务时，才调用 Motive 生成动机 → MainPlanner 规划
"""
import asyncio
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from src.common.logger import get_logger
from src.common.di.container import container
from src.core.task.models import Priority, TaskStatus
from src.core.task.task_store import TaskStore
from src.core.task.task_manager import TaskManager
from src.core.kernel.scheduler import Scheduler
from src.core.kernel.interrupt_handler import InterruptHandler
from src.agent.motive.motive_engine import MotiveEngine
from src.agent.planner.main_planner import MainPlanner
from src.agent.world_model import WorldModel

logger = get_logger("event_loop")


@dataclass
class InterruptSignal:
    """中断信号"""
    source_cortex: str
    target_id: str
    content: str
    priority: Priority
    raw_data: Dict[str, Any]
    timestamp: float


class EventLoop:
    """
    事件驱动核心循环

    替代旧的 AgentLoop，实现真正的内核事件驱动模式
    """

    def __init__(self):
        # 核心组件
        self.task_manager: TaskManager = container.resolve(TaskManager)
        self.scheduler: Scheduler = container.resolve(Scheduler)
        self.interrupt_handler: InterruptHandler = container.resolve(InterruptHandler)
        self.task_store: TaskStore = container.resolve(TaskStore)

        # 认知组件
        self.motive_engine: MotiveEngine = MotiveEngine()
        self.main_planner: MainPlanner = MainPlanner()
        self.world_model: WorldModel = container.resolve(WorldModel)

        # 事件队列
        self._interrupt_queue: asyncio.Queue[InterruptSignal] = asyncio.Queue(maxsize=100)
        self._is_running = False
        self._main_event_task: Optional[asyncio.Task] = None

        # 回调钩子
        self._on_planning: Optional[Callable] = None
        self.__on_execution: Optional[Callable] = None

    async def start(self):
        """启动事件循环"""
        if self._is_running:
            logger.warning("EventLoop 已经在运行中")
            return

        self._is_running = True
        logger.info("事件驱动核心循环已启动")

        # 启动事件处理任务
        self._main_event_task = asyncio.create_task(self._run_event_loop())

        # 启动空闲调度任务（当无任务时触发）
        asyncio.create_task(self._idle_scheduler())

    async def stop(self):
        """停止事件循环"""
        if not self._is_running:
            return

        self._is_running = False

        if self._main_event_task:
            self._main_event_task.cancel()

        logger.info("事件驱动核心循环已停止")

    async def submit_interrupt(self, signal: InterruptSignal):
        """提交中断信号"""
        await self._interrupt_queue.put(signal)
        logger.debug(f"提交中断信号: {signal.source_cortex} -> {signal.target_id}")

    async def _run_event_loop(self):
        """主事件循环 - 处理中断信号"""
        while self._is_running:
            try:
                # 等待中断信号
                signal = await asyncio.wait_for(
                    self._interrupt_queue.get(),
                    timeout=1.0
                )

                logger.info(f"📥 处理中断: {signal.source_cortex} -> {signal.target_id} (优先级: {signal.priority.value})")

                # 1. 创建/复用任务
                task = await self.task_manager.create_task(
                    cortex=signal.source_cortex,
                    target_id=signal.target_id,
                    priority=signal.priority,
                    motive=f"处理来自 {signal.source_cortex} 的信号"
                )

                # 2. 调度器选择焦点任务
                focus_task_id = await self.scheduler.schedule(force=True)

                if focus_task_id:
                    logger.info(f"焦点任务已调度: {focus_task_id}")
                    # 这里可以触发任务执行（如果需要）

            except asyncio.TimeoutError:
                # 队列为空，继续等待
                continue
            except asyncio.CancelledError:
                logger.info("事件循环任务被取消")
                break
            except Exception as e:
                logger.error(f"处理中断信号失败: {e}", exc_info=True)

    async def _idle_scheduler(self):
        """
        空闲调度器

        只有当真正没有任何需要处理的任务时，才调用 Motive 生成动机。

        注意：BACKGROUND 监听任务不算"空闲"，因为它们在等待外部事件
        """
        logger.info("空闲调度器已启动")

        while self._is_running:
            try:
                # 检查是否有需要处理的任务
                focus_task = await self.task_store.get_focus_task()

                # 只检查 READY/BACKGROUND 状态的任务（不包括 MUTED）
                ready_tasks = await self.task_store.list_all(status=TaskStatus.READY)
                background_tasks = await self.task_store.list_all(status=TaskStatus.BACKGROUND)

                # 只有当没有焦点任务且没有就绪任务时，才调用 Motive
                # BACKGROUND 监听任务不计入，因为它们在被动等待
                if not focus_task and not ready_tasks and not background_tasks:
                    logger.debug("系统真正空闲（无FOCUS,READY,BACKGROUND任务），调用 Motive 生成动机")

                    # 1. 生成动机
                    motive = await self.motive_engine.generate_motive()

                    if motive:
                        logger.info(f"🎯 生成动机: {motive}")

                        # 2. 刷新任务快照
                        await self.world_model.refresh_task_snapshots()

                        # 3. 调用规划器
                        if self._on_planning:
                            await self._on_planning()

                        shell_plan = await self.main_planner.plan(
                            motive,
                            self.world_model.get_last_observation()
                        )

                        # 4. 执行 Shell 指令
                        if shell_plan:
                            from src.core.kernel.interpreter import KernelInterpreter
                            interpreter = container.resolve(KernelInterpreter)

                            results = await interpreter.execute_batch(shell_plan)
                            logger.info(f"✅ 执行了 {len(results)} 条指令")

                            # 更新观察
                            import json
                            observation = json.dumps(results, ensure_ascii=False)
                            self.world_model.set_last_observation(observation)

                    # 5. 等待一段时间再检查
                    await asyncio.sleep(5)
                else:
                    # 有任务在运行，短暂等待
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("空闲调度器被取消")
                break
            except Exception as e:
                logger.error(f"空闲调度器失败: {e}", exc_info=True)
                await asyncio.sleep(5)

    def set_planning_hook(self, hook: Callable):
        """设置规划过程钩子"""
        self._on_planning = hook

    def set_execution_hook(self, hook: Callable):
        """设置执行过程钩子"""
        self._on_execution = hook

    async def get_system_state(self) -> Dict[str, Any]:
        """获取系统状态"""
        focus_task = await self.task_store.get_focus_task()
        ready_tasks = await self.task_store.list_ready_by_priority()

        return {
            "is_running": self._is_running,
            "interrupt_queue_size": self._interrupt_queue.qsize(),
            "has_focus_task": focus_task is not None,
            "focus_task_id": focus_task.task_id if focus_task else None,
            "ready_task_count": len(ready_tasks),
        }


# 单例实例
_event_loop_instance: Optional[EventLoop] = None


def get_event_loop() -> EventLoop:
    """获取 EventLoop 单例"""
    global _event_loop_instance
    if _event_loop_instance is None:
        _event_loop_instance = EventLoop()
    return _event_loop_instance


def clear_event_loop():
    """清除 EventLoop 单例"""
    global _event_loop_instance
    if _event_loop_instance is not None:
        _event_loop_instance = None
