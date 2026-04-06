import time
from typing import Any, Dict, Optional

from src.common.logger import get_logger
from src.core.task.models import Task, TaskMode, TaskStatus, Priority
from src.core.task.task_store import TaskStore, _PRIORITY_ORDER

logger = get_logger("task_manager")


class TaskManager:
    """
    进程管理器
    - 负责执行 Task 类中定义的架构语义：唯一性判定、优先级继承、被动挂起、抢占与复活。
    """

    def __init__(self, task_store: TaskStore):
        self.task_store = task_store

    def _should_preempt(self, current: Optional[Task], incoming: Task) -> bool:
        """
        判断是否需要对当前 FOCUS 进行抢占。
        逻辑：只要新任务优先级高于当前焦点，就触发抢占。
        """
        if not current:
            return False
        return _PRIORITY_ORDER.get(incoming.priority, 0) > _PRIORITY_ORDER.get(current.priority, 0)

    def _default_status_for_mode(self, mode: TaskMode) -> TaskStatus:
        """
        为不同 task mode 提供默认启动状态。
        """
        if mode in {TaskMode.LISTEN, TaskMode.CRON}:
            return TaskStatus.BACKGROUND
        return TaskStatus.READY

    # ---------- 核心接口 ----------
    async def create_task(
        self,
        cortex: str,
        target_id: str,
        priority: Priority = Priority.LOW,
        motive: str = "",
        mode: TaskMode = TaskMode.ONCE,
        context_ref: str = "",
        task_config: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        创建或复用任务环境 (Process Environment)：
        - 确保物理载体 (target_id) 与 意识源 (cortex) 对应的 Task 实例存在。
        - 仅负责环境初始化、模式配置、优先级同步与状态唤醒。
        - 新架构中 Task 不再默认绑定 action 栈，而是记录当前任务窗口与运行模式。
        """
        # 1. 寻找现有环境
        existing = await self.task_store.get_by_target(target_id, cortex)
        focus_task = await self.task_store.get_focus_task()
        normalized_config = task_config or {}
        
        if existing:
            task = existing[0]
            # --- 环境复用逻辑 ---

            task.priority = priority if _PRIORITY_ORDER.get(priority, 0) >= _PRIORITY_ORDER.get(task.priority, 0) else task.priority

            # 唤醒逻辑：如果进程处于休眠或静默态，重新标记为 READY
            if task.status in {TaskStatus.MUTED, TaskStatus.SUSPENDED, TaskStatus.BLOCKED}:
                task.status = self._default_status_for_mode(mode)
            elif task.status == TaskStatus.BACKGROUND and mode not in {TaskMode.LISTEN, TaskMode.CRON}:
                task.status = TaskStatus.READY
            
            # 更新环境锚点 (Metadata Only)
            if motive:
                task.motive = motive
            if context_ref:
                task.context_ref = context_ref
            task.mode = mode
            if normalized_config:
                task.task_config.update(normalized_config)
            task.updated_at = time.time()
            
            await self.task_store.save(task)
            logger.info(f"复用并唤醒任务环境: id={task.task_id} target={target_id}")
        else:
            # --- 环境新建逻辑 ---
            task = Task(
                target_id=target_id,
                cortex=cortex,
                priority=priority,
                status=self._default_status_for_mode(mode),
                mode=mode,
                motive=motive,
                context_ref=context_ref,
                task_config=normalized_config,
            )
            await self.task_store.save(task)
            logger.info(f"新建任务环境: id={task.task_id} target={target_id}")

        # 2. 焦点调度逻辑 (Kernel Scheduler)
        # 判断当前新唤醒/创建的环境是否应该立即抢占 CPU (Focus)
        preempt = self._should_preempt(focus_task, task)
        
        if task.mode in {TaskMode.LISTEN, TaskMode.CRON}:
            # 监听/定时任务默认留在后台，不主动抢占。
            await self.task_store.save(task)
        elif preempt:
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
    
    async def inject_context(self, target_id: str, cortex: str, context_data: Dict[str, Any], mode: TaskMode = TaskMode.ONCE) -> None:
        """
        向任务窗口静默注入上下文。
        """
        # 1. 找到或创建环境 (Task)
        task = await self.create_task(target_id=target_id, cortex=cortex, mode=mode)
        
        # 2. 注入到任务窗口中，而不是旧 action 栈。
        role = context_data.get("role", "observation")
        content = context_data.get("content", "")
        metadata = context_data.get("metadata", {})
        task.append_window_message(role=role, content=content, **metadata)
        
        # 3. 保持任务自身状态，不主动抢占。
        await self.task_store.save(task)

    async def get_focus_task(self) -> Optional[Task]:
        """读取当前焦点任务。"""
        return await self.task_store.get_focus_task()

    async def update_task_runtime(
        self,
        task_id: str,
        *,
        last_observation: Optional[str] = None,
        last_result: Optional[Dict[str, Any]] = None,
        last_signal: Optional[Dict[str, Any]] = None,
        append_window: Optional[Dict[str, Any]] = None,
        increment_execution: bool = False,
    ) -> Optional[Task]:
        """
        更新任务运行态。
        """
        task = await self.task_store.get(task_id)
        if not task:
            return None

        if last_observation is not None:
            task.last_observation = last_observation
        if last_result is not None:
            task.set_last_result(last_result)
        if last_signal is not None:
            task.set_last_signal(last_signal)
        if append_window:
            task.append_window_message(
                role=append_window.get("role", "observation"),
                content=append_window.get("content", ""),
                **(append_window.get("metadata", {}) or {}),
            )
        if increment_execution:
            task.execution_count += 1

        await self.task_store.save(task)
        return task
