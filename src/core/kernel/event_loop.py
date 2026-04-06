"""
Event Loop - 事件驱动核心循环

新版职责：
1. 统一接收中断信号、空闲动机、调试台输入。
2. 优先把高频 QQ 消息交给回复器监听任务处理。
3. 只在需要时调用主 Planner，并执行其 JSON 规划结果。
"""
import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from src.agent.motive.motive_engine import MotiveEngine
from src.agent.planner.main_planner import MainPlanner
from src.agent.planner.planner_result import PlannerResult
from src.agent.world_model import WorldModel
from src.common.di.container import container
from src.common.logger import get_logger
from src.core.kernel.interrupt_handler import InterruptHandler
from src.core.kernel.scheduler import Scheduler
from src.core.reply import ReplyRuntime
from src.core.task.models import Priority, TaskMode, TaskStatus
from src.core.task.task_manager import TaskManager
from src.core.task.task_store import TaskStore

logger = get_logger("event_loop")


@dataclass
class InterruptSignal:
    source_cortex: str
    target_id: str
    content: str
    priority: Priority
    raw_data: Dict[str, Any]
    timestamp: float


class EventLoop:
    def __init__(self):
        self.task_manager: TaskManager = container.resolve(TaskManager)
        self.scheduler: Scheduler = container.resolve(Scheduler)
        self.interrupt_handler: InterruptHandler = container.resolve(InterruptHandler)
        self.task_store: TaskStore = container.resolve(TaskStore)

        self.motive_engine: MotiveEngine = MotiveEngine()
        self.main_planner: MainPlanner = MainPlanner()
        self.reply_runtime = ReplyRuntime()
        self.world_model: WorldModel = container.resolve(WorldModel)

        self._interrupt_queue: asyncio.Queue[InterruptSignal] = asyncio.Queue(maxsize=100)
        self._debug_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=20)
        self._is_running = False
        self._main_event_task: Optional[asyncio.Task] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._debug_task: Optional[asyncio.Task] = None

        self._on_planning: Optional[Callable] = None
        self._on_execution: Optional[Callable] = None

    async def start(self):
        if self._is_running:
            logger.warning("EventLoop 已经在运行中")
            return

        self._is_running = True
        logger.info("事件驱动核心循环已启动")
        self._main_event_task = asyncio.create_task(self._run_event_loop())
        self._idle_task = asyncio.create_task(self._idle_scheduler())
        self._debug_task = asyncio.create_task(self._run_debug_loop())

    async def stop(self):
        if not self._is_running:
            return

        self._is_running = False
        for task in (self._main_event_task, self._idle_task, self._debug_task):
            if task:
                task.cancel()

        logger.info("事件驱动核心循环已停止")

    async def submit_interrupt(self, signal: InterruptSignal):
        await self._interrupt_queue.put(signal)
        logger.debug(f"提交中断信号: {signal.source_cortex} -> {signal.target_id}")

    async def submit_debug_input(self, content: str):
        await self._debug_queue.put(content)
        logger.info(f"收到调试台输入: {content[:80]}")

    async def _run_event_loop(self):
        while self._is_running:
            try:
                signal = await asyncio.wait_for(self._interrupt_queue.get(), timeout=1.0)
                logger.info(f"📥 处理中断: {signal.source_cortex} -> {signal.target_id} (优先级: {signal.priority.value})")
                await self._handle_interrupt_signal(signal)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("事件循环任务被取消")
                break
            except Exception as exc:
                logger.error(f"处理中断信号失败: {exc}", exc_info=True)

    async def _run_debug_loop(self):
        while self._is_running:
            try:
                debug_request = await asyncio.wait_for(self._debug_queue.get(), timeout=1.0)
                await self._plan_and_execute(
                    input_source="debug_input",
                    motive="响应开发者调试台请求",
                    latest_signal={},
                    debug_request=debug_request,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("调试输入循环被取消")
                break
            except Exception as exc:
                logger.error(f"调试输入处理失败: {exc}", exc_info=True)

    async def _handle_interrupt_signal(self, signal: InterruptSignal):
        normalized_cortex = self._normalize_cortex_name(signal.source_cortex)
        signal_payload = {
            "source_cortex": normalized_cortex,
            "target_id": signal.target_id,
            "content": signal.content,
            "priority": signal.priority.value,
            "raw_data": signal.raw_data,
            "timestamp": signal.timestamp,
        }

        # 先确保相关任务存在。
        task = await self.task_manager.create_task(
            cortex=normalized_cortex,
            target_id=signal.target_id,
            priority=signal.priority,
            motive=f"处理来自 {normalized_cortex} 的中断信号",
            mode=TaskMode.ONCE,
            task_config={"input_source": "interrupt_input"},
        )
        await self.task_manager.update_task_runtime(
            task.task_id,
            last_signal=signal_payload,
            append_window={
                "role": "observation",
                "content": signal.content,
                "metadata": {"source_cortex": normalized_cortex},
            },
        )

        # 高频 QQ 消息优先交给回复器，不必每次都唤醒主 Planner。
        reply_result = await self.reply_runtime.handle_signal(signal)
        if reply_result.handled and not reply_result.escalated:
            logger.info(f"回复器已接管该信号: {reply_result.reason}")
            return

        await self._plan_and_execute(
            input_source="interrupt_input",
            motive=f"处理中断信号：{signal.content}",
            latest_signal=signal_payload,
        )

    def _normalize_cortex_name(self, source_cortex: str) -> str:
        normalized = str(source_cortex).lower()
        if normalized in {"qq", "qqchat", "qq_chat"}:
            return "qq_chat"
        return normalized

    async def _idle_scheduler(self):
        logger.info("空闲调度器已启动")

        while self._is_running:
            try:
                focus_task = await self.task_store.get_focus_task()
                ready_tasks = await self.task_store.list_all(status=TaskStatus.READY)

                if not focus_task and not ready_tasks:
                    logger.debug("系统空闲（无 FOCUS、无 READY 任务），调用 Motive 生成动机")
                    motive = await self.motive_engine.generate_motive()
                    if motive:
                        logger.info(f"🎯 生成动机: {motive}")
                        await self.world_model.refresh_task_snapshots()
                        await self._plan_and_execute(
                            input_source="idle_input",
                            motive=motive,
                            latest_signal={},
                        )

                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("空闲调度器被取消")
                break
            except Exception as exc:
                logger.error(f"空闲调度器失败: {exc}", exc_info=True)
                await asyncio.sleep(5)

    async def _plan_and_execute(
        self,
        *,
        input_source: str,
        motive: str,
        latest_signal: Dict[str, Any],
        debug_request: str = "",
    ) -> PlannerResult:
        """
        统一调用 Planner 并执行其输出。
        """
        if self._on_planning:
            await self._on_planning()

        planner_result = await self.main_planner.plan(
            motive=motive,
            previous_observation=self.world_model.get_last_observation(),
            input_source=input_source,
            latest_signal=latest_signal,
            debug_request=debug_request,
        )

        from src.core.kernel.interpreter import KernelInterpreter

        interpreter = container.resolve(KernelInterpreter)
        execution_results = await interpreter.execute_batch(planner_result)

        if self._on_execution:
            await self._on_execution()

        observation = json.dumps(
            {
                "thought": planner_result.thought,
                "results": execution_results,
            },
            ensure_ascii=False,
        )
        self.world_model.set_last_observation(observation)
        return planner_result

    def set_planning_hook(self, hook: Callable):
        self._on_planning = hook

    def set_execution_hook(self, hook: Callable):
        self._on_execution = hook

    async def get_system_state(self) -> Dict[str, Any]:
        focus_task = await self.task_store.get_focus_task()
        ready_tasks = await self.task_store.list_ready_by_priority()
        return {
            "is_running": self._is_running,
            "interrupt_queue_size": self._interrupt_queue.qsize(),
            "debug_queue_size": self._debug_queue.qsize(),
            "has_focus_task": focus_task is not None,
            "focus_task_id": focus_task.task_id if focus_task else None,
            "ready_task_count": len(ready_tasks),
        }


_event_loop_instance: Optional[EventLoop] = None


def get_event_loop() -> EventLoop:
    global _event_loop_instance
    if _event_loop_instance is None:
        _event_loop_instance = EventLoop()
    return _event_loop_instance


def clear_event_loop():
    global _event_loop_instance
    if _event_loop_instance is not None:
        _event_loop_instance = None
