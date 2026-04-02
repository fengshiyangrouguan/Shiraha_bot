from typing import Optional, Any, Dict

from src.common.logger import get_logger
from src.common.di.container import container
from src.core.task.models import Priority, TaskStatus
from src.core.task.task_store import TaskStore, _PRIORITY_ORDER

logger = get_logger("kernel_scheduler")


class Scheduler:
    """
    内核调度器 (Kernel Scheduler)
    直接基于 TaskStore 进行调度，避免维护第二套 ready_queue。
    """

    def __init__(self):
        self.task_store: TaskStore = container.resolve(TaskStore)
        self.current_task_id: Optional[str] = None

    def _priority_rank(self, priority: Priority) -> int:
        return _PRIORITY_ORDER.get(priority, 0)

    async def schedule(self, force: bool = False) -> Optional[str]:
        """
        选择下一焦点任务：
        1. 默认保持当前焦点（除非 force=True）。
        2. 优先从 READY/BACKGROUND/MUTED 中挑最高优先级任务。
        3. 若无可执行任务，尝试复活一个 SUSPENDED。
        """
        current_focus = await self.task_store.get_focus_task()
        if current_focus and not force:
            self.current_task_id = current_focus.task_id
            return current_focus.task_id

        ready_tasks = await self.task_store.list_ready_by_priority()
        if ready_tasks:
            next_task = ready_tasks[0]
            await self.task_store.set_focus(next_task.task_id, preempt=bool(current_focus and current_focus.task_id != next_task.task_id))
            self.current_task_id = next_task.task_id
            return next_task.task_id

        revived = await self.task_store.revive_one_suspended()
        if revived:
            await self.task_store.set_focus(revived.task_id)
            self.current_task_id = revived.task_id
            return revived.task_id

        self.current_task_id = None
        return None

    async def switch_context(self, next_task_id: str, preempt: bool = False) -> None:
        """
        执行上下文切换：由 TaskStore 维护唯一焦点与被动挂起语义。
        """
        if self.current_task_id == next_task_id:
            return

        logger.info(f"Context Switch: {self.current_task_id} -> {next_task_id} (preempt={preempt})")
        await self.task_store.set_focus(next_task_id, preempt=preempt)
        self.current_task_id = self.task_store.current_focus()

    async def add_to_ready(self, task_id: str, priority: Optional[Priority] = None) -> Optional[str]:
        """
        将任务放入 READY 状态；如果提供优先级则一并更新。
        """
        task = await self.task_store.get(task_id)
        if not task:
            logger.warning(f"add_to_ready 失败，任务不存在: {task_id}")
            return None

        if priority is not None:
            task.priority = priority
        task.status = TaskStatus.READY
        await self.task_store.save(task)
        return task.task_id

    async def dispatch_to_executor(self, task_id: str, entry: Optional[str]) -> Dict[str, Any]:
        """
        分发执行：先切换焦点，再交给执行器（当前阶段仅返回调度结果，执行器后续接入）。
        """
        task = await self.task_store.get(task_id)
        if not task:
            raise ValueError(f"任务不存在: {task_id}")

        current_focus = await self.task_store.get_focus_task()
        should_preempt = bool(
            current_focus
            and current_focus.task_id != task.task_id
            and self._priority_rank(task.priority) > self._priority_rank(current_focus.priority)
        )
        await self.switch_context(task.task_id, preempt=should_preempt)

        logger.info(f"Dispatching Task {task_id} to Entry: {entry or 'default'}")
        return {
            "task_id": task_id,
            "entry": entry or "default",
            "status": "dispatched",
            "preempt": should_preempt,
        }

    async def handle_interrupt(self, interrupt_task_id: str, priority: Priority) -> None:
        """
        抢占式中断处理：新任务优先级更高时抢占，否则放回 READY。
        """
        interrupt_task = await self.task_store.get(interrupt_task_id)
        if not interrupt_task:
            logger.warning(f"中断任务不存在: {interrupt_task_id}")
            return

        current_focus = await self.task_store.get_focus_task()
        current_priority = current_focus.priority if current_focus else Priority.LOW

        if self._priority_rank(priority) > self._priority_rank(current_priority):
            logger.warning(
                f"Preemptive Interrupt: new={interrupt_task_id}({priority.value}) > current({current_priority.value})"
            )
            await self.switch_context(interrupt_task_id, preempt=bool(current_focus and current_focus.task_id != interrupt_task_id))
            return

        await self.add_to_ready(interrupt_task_id, priority=priority)
