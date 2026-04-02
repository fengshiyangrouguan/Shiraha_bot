import time
from typing import Optional, Iterable

from src.common.logger import get_logger
from src.core.task.models import Task, TaskStatus, Priority, BaseAction
from src.core.task.task_store import TaskStore, _PRIORITY_ORDER # TaskStore 中定义的优先级权重映射

logger = get_logger("task_manager")


class TaskManager:
    """
    进程管理器
    - 负责执行 Task 类中定义的架构语义：唯一性判定、优先级继承、被动挂起、抢占与复活。
    """

    def __init__(self, task_store: TaskStore):
        self.task_store = task_store

    # ---------- 工具方法 ----------
    def _auto_set_priority(self, actions: Iterable[BaseAction], base: Priority) -> Priority:
        """
        根据 Task 中的“优先级继承”原则：Task 优先级 = max(所有 Action.priority，task本身的优先级)。
        现在 Action.priority 已统一为 Priority 枚举，这里直接比较枚举权重。
        """
        highest: Optional[Priority] = None
        for act in actions or []:
            derived = act.priority
            if highest is None or _PRIORITY_ORDER[derived] > _PRIORITY_ORDER[highest]:
                highest = derived
        return highest or base

    def _should_preempt(self, current: Optional[Task], incoming: Task) -> bool:
        """
        判断是否需要对当前 FOCUS 进行抢占。
        逻辑：只要新任务优先级高于当前焦点，就触发抢占。
        """
        if not current:
            return False
        return _PRIORITY_ORDER.get(incoming.priority, 0) > _PRIORITY_ORDER.get(current.priority, 0)

    # ---------- 核心接口 ----------
    async def create_task(
        self,
        cortex: str,
        target_id: str,
        priority: Priority = Priority.LOW,
        motive: str = "",
        context_ref: str = "",
    ) -> Task:
        """
        创建或复用任务环境 (Process Environment)：
        - 确保物理载体 (target_id) 与 意识源 (cortex) 对应的 Task 实例存在。
        - 仅负责环境初始化、优先级同步与状态唤醒。
        - 具体的业务行为由后续的 push_action 完成。
        """
        # 1. 寻找现有环境
        existing = await self.task_store.get_by_target(target_id, cortex)
        focus_task = await self.task_store.get_focus_task()
        
        if existing:
            task = existing[0]
            # --- 环境复用逻辑 ---
            
            # 优先级同步：取当前行为栈最高优先级与传入优先级的最大值
            # 确保即使没有新 Action，Task 的权位也能被正确抬升
            current_max_p = self._auto_set_priority(task.actions, priority)
            task.priority = current_max_p if _PRIORITY_ORDER.get(current_max_p, 0) > _PRIORITY_ORDER.get(priority, 0) else priority

            # 唤醒逻辑：如果进程处于休眠或静默态，重新标记为 READY
            if task.status in {TaskStatus.BACKGROUND, TaskStatus.MUTED, TaskStatus.SUSPENDED}:
                task.status = TaskStatus.READY
            
            # 更新环境锚点 (Metadata Only)
            if motive: task.motive = motive
            if context_ref: task.context_ref = context_ref
            task.updated_at = time.time()
            
            await self.task_store.save(task)
            logger.info(f"复用并唤醒任务环境: id={task.task_id} target={target_id}")
        else:
            # --- 环境新建逻辑 ---
            task = Task(
                target_id=target_id,
                cortex=cortex,
                priority=priority,
                status=TaskStatus.READY,
                motive=motive,
                context_ref=context_ref,
            )
            await self.task_store.save(task)
            logger.info(f"新建任务环境: id={task.task_id} target={target_id}")

        # 2. 焦点调度逻辑 (Kernel Scheduler)
        # 判断当前新唤醒/创建的环境是否应该立即抢占 CPU (Focus)
        preempt = self._should_preempt(focus_task, task)
        
        if preempt:
            # 触发抢占：旧 Task 会被内核自动变为 SUSPENDED
            await self.task_store.set_focus(task.task_id, preempt=True)
        elif not focus_task:
            # 无焦点时直接占位
            await self.task_store.set_focus(task.task_id)

        return task

    async def suspend_task(self, task_id: str) -> Optional[Task]:
        """
        主动挂起任务（进入 SUSPENDED）。
        """
        task = await self.task_store.update_status(task_id, TaskStatus.SUSPENDED)
        if task:
            logger.info(f"任务已挂起: {task.task_id}")
        return task

    async def resume_task(self, task_id: str) -> Optional[Task]:
        """
        将 SUSPENDED/BLOCKED/MUTED/BACKGROUND 任务恢复为 READY。
        若其优先级高于当前焦点，则触发抢占。
        """
        task = await self.task_store.get(task_id)
        if not task:
            return None

        task.status = TaskStatus.READY
        await self.task_store.save(task)

        focus_task = await self.task_store.get_focus_task()
        if self._should_preempt(focus_task, task):
            await self.task_store.set_focus(task.task_id, preempt=True)
        elif not focus_task:
            await self.task_store.set_focus(task.task_id)

        logger.info(f"任务已恢复为 READY: {task.task_id}")
        return task

    async def block_task(self, task_id: str) -> Optional[Task]:
        """
        将任务置为 BLOCKED，用于等待外部事件或工具回调。
        """
        task = await self.task_store.update_status(task_id, TaskStatus.BLOCKED)
        if task:
            logger.info(f"任务进入阻塞态: {task.task_id}")
        return task

    async def terminate_task(self, task_id: str) -> None:
        """
        终结任务生命周期，释放索引与焦点占用。
        """
        task = await self.task_store.update_status(task_id, TaskStatus.TERMINATED)
        if task:
            logger.info(f"任务已终结: {task.task_id}")

    async def adjust_priority(self, task_id: str, priority: Priority) -> Optional[Task]:
        """
        动态调整任务优先级；若高于当前焦点则立即抢占。
        """
        task = await self.task_store.get(task_id)
        if not task:
            return None

        task.priority = priority
        await self.task_store.save(task)

        focus_task = await self.task_store.get_focus_task()
        if self._should_preempt(focus_task, task):
            await self.task_store.set_focus(task.task_id, preempt=True)

        logger.info(f"任务优先级已调整: {task.task_id} -> {task.priority.value}")
        return task
    
    async def inject_context(self, target_id: str, cortex: str, context_data: dict) -> None:
        # TODO:接口设计不稳定，得改一下context_data的数据格式，实际是个 List[Dict]
        # 1. 找到或创建环境 (Task)
        task = await self.create_task(target_id=target_id, cortex=cortex)
        
        # 2. 注入数据到“环境缓冲区”，而不是“执行栈”
        # 这里的数据是“静默”的，直到下一个 Action 被唤醒时才会读取它
        task.update_context_buffer(context_data)
        
        # 3. 保持状态：如果任务是 BACKGROUND，注入后依然是 BACKGROUND
        # 它不需要抢占 FOCUS，因为它没有活儿要干
        await self.task_store.save(task)

    async def get_focus_task(self) -> Optional[Task]:
        """读取当前焦点任务。"""
        return await self.task_store.get_focus_task()
