import time
from typing import Optional, List
from src.core.task.models import Task, TaskStatus
from src.core.task.task_store import TaskStore
from src.common.logger import get_logger

logger = get_logger("task_manager")

class TaskManager:
    """
    任务管理器 (Task Lifecycle Manager)
    实现任务的 CRUD 和 状态流转逻辑。
    """
    def __init__(self, store: TaskStore):
        self.store = store

    async def create_task(self, cortex: str, target_id: str, priority: int, motive: str) -> Task:
        """创建一个新进程"""
        # 检查是否已存在同目标的活跃任务，实现单目标任务互斥
        existing = await self.store.get_by_target(target_id)
        if existing:
            logger.warning(f"Target {target_id} already has active tasks: {[t.task_id for t in existing]}")
            # 这里可以根据策略选择合并任务或返回既有任务

        new_task = Task(
            cortex=cortex,
            target_id=target_id,
            priority=priority,
            motive=motive,
            context_ref=f"ctx_{target_id}_{int(time.time())}" # 自动生成上下文引用
        )
        await self.store.save(new_task)
        logger.info(f"🆕 Task Created: {new_task.task_id} for {target_id}")
        return new_task

    async def suspend_task(self, task_id: str):
        """挂起进程：对应指令 task suspend"""
        task = await self.store.get(task_id)
        if task:
            task.status = TaskStatus.SUSPENDED
            await self.store.save(task)
            logger.info(f"⏸️ Task Suspended: {task_id}")

    async def resume_task(self, task_id: str):
        """恢复进程：对应指令 task resume"""
        task = await self.store.get(task_id)
        if task:
            task.status = TaskStatus.READY # 回到就绪队列等待调度器拾取
            await self.store.save(task)
            logger.info(f"▶️ Task Resumed: {task_id}")

    async def terminate_task(self, task_id: str):
        """销毁进程：对应指令 task kill"""
        task = await self.store.get(task_id)
        if task:
            task.status = TaskStatus.TERMINATED
            await self.store.save(task)
            # 这里可以执行一些清理工作，比如触发总结存库
            logger.info(f"💀 Task Terminated: {task_id}")
