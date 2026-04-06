from typing import Dict, Optional, List, Tuple
import time

from src.core.task.models import Task, TaskStatus, Priority
from src.common.logger import get_logger

logger = get_logger("task_store")

# 优先级权重，用于排序（数字越大越重要）
_PRIORITY_ORDER = {
    Priority.CRITICAL: 3,
    Priority.HIGH: 2,
    Priority.MEDIUM: 1,
    Priority.LOW: 0,
}


class TaskStore:
    """
    任务状态存储
    - 负责 Task 的生命周期持久化与唯一性判定（target_id + cortex）。
    - 后续可以无缝替换为 Redis/SQLite，接口保持兼容。
    - 严格对齐 Task 中定义的架构语义：READY / FOCUS / SUSPENDED / BLOCKED / MUTED / BACKGROUND / TERMINATED。
    """

    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._focus_task_id: Optional[str] = None
        # (target_id, cortex) -> task_id，用于 O(1) 唯一性查重，加速 get_by_target
        self._target_index: Dict[Tuple[str, str], str] = {}
        # 记录被动挂起（SUSPENDED）的 task_id，维持“优先复活权”队列（先进先出）
        self._suspended_stack: List[str] = []

    # ---------- CRUD ----------
    async def save(self, task: Task) -> Task:
        """
        保存 / 覆写 Task，同时刷新更新时间与唯一性索引。
        这里只关注幂等写入，不做业务判定。
        """
        task.updated_at = time.time()
        self._tasks[task.task_id] = task
        key = (task.target_id, task.cortex)
        if task.status != TaskStatus.TERMINATED:
            self._target_index[key] = task.task_id
        else:
            self._target_index.pop(key, None)
        return task

    async def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    async def delete(self, task_id: str) -> None:
        """彻底删除任务的内存快照，同时清理索引与挂起队列。"""
        task = self._tasks.pop(task_id, None)
        if task:
            self._target_index.pop((task.target_id, task.cortex), None)
            if task_id in self._suspended_stack:
                self._suspended_stack = [tid for tid in self._suspended_stack if tid != task_id]
        if self._focus_task_id == task_id:
            self._focus_task_id = None

    # ---------- Queries ----------
    async def get_by_target(self, target_id: str, cortex: Optional[str] = None) -> List[Task]:
        """
        基于 (target_id, cortex) 做唯一性查找。
        - cortex 为空时返回同 target_id 下所有未终止任务。
        - cortex 不为空时优先走索引命中，避免线性扫描。
        """
        if cortex is not None:
            mapped_id = self._target_index.get((target_id, cortex))
            candidate = self._tasks.get(mapped_id) if mapped_id else None
            if candidate and candidate.status != TaskStatus.TERMINATED:
                return [candidate]

        return [
            t for t in self._tasks.values()
            if t.target_id == target_id
            and (cortex is None or t.cortex == cortex)
            and t.status != TaskStatus.TERMINATED
        ]

    async def list_all(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """
        获取全部任务，或按状态过滤。
        """
        if status:
            return [t for t in self._tasks.values() if t.status == status]
        return list(self._tasks.values())

    async def list_ready_by_priority(self) -> List[Task]:
        """
        返回按优先级降序、创建时间升序的 READY/BACKGROUND/MUTED 任务列表。
        FOCUS/SUSPENDED/BLOCKED 会被过滤，符合 Task 架构中“只有 READY/BACKGROUND/MUTED 参与抢占”的设计。
        """
        buckets = {Priority.CRITICAL: [], Priority.HIGH: [], Priority.MEDIUM: [], Priority.LOW: []}
        for t in self._tasks.values():
            if t.status in {TaskStatus.READY, TaskStatus.BACKGROUND, TaskStatus.MUTED}:
                buckets[t.priority].append(t)
        ordered: List[Task] = []
        for level in (Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW):
            ordered.extend(sorted(buckets[level], key=lambda x: x.created_at))
        return ordered

    async def list_suspended(self) -> List[Task]:
        """
        返回当前处于 SUSPENDED 的任务列表，优先级高者在前，其次按创建时间。
        结合 Task 中的“优先复活权”，调度器在高优抢占结束后应优先恢复这里的任务。
        """
        candidates = [
            self._tasks[tid]
            for tid in self._suspended_stack
            if tid in self._tasks and self._tasks[tid].status == TaskStatus.SUSPENDED
        ]
        return sorted(
            candidates,
            key=lambda t: (-_PRIORITY_ORDER.get(t.priority, 0), t.created_at)
        )

    async def revive_one_suspended(self) -> Optional[Task]:
        """
        弹出“最该被恢复”的挂起任务并置为 READY，便于调度器重新排队。
        """
        ordered = await self.list_suspended()
        if not ordered:
            return None
        candidate = ordered[0]
        # 从挂起队列移除并恢复为 READY
        self._suspended_stack = [tid for tid in self._suspended_stack if tid != candidate.task_id]
        candidate.status = TaskStatus.READY
        await self.save(candidate)
        return candidate

    # ---------- Status / Focus helpers ----------
    async def update_status(self, task_id: str, status: TaskStatus) -> Optional[Task]:
        task = await self.get(task_id)
        if not task:
            return None

        previous_status = task.status
        task.status = status
        await self.save(task)

        # 聚焦处理：保持全局唯一焦点
        if status == TaskStatus.FOCUS:
            await self.set_focus(task_id)
        elif status == TaskStatus.TERMINATED:
            if self._focus_task_id == task_id:
                self._focus_task_id = None
            await self.delete(task_id)
            return None

        # 维护 SUSPEND <-> 其它状态的队列一致性
        if status == TaskStatus.SUSPENDED:
            self._push_suspended(task_id)
        elif previous_status == TaskStatus.SUSPENDED and status != TaskStatus.SUSPENDED:
            self._remove_suspended(task_id)

        return task

    async def set_focus(self, task_id: Optional[str], preempt: bool = False) -> None:
        """
        确保系统同一时刻只有一个 FOCUS 任务。
        preempt=True 表示发生了抢占：原焦点被送入 SUSPENDED（被动挂起，享受“优先复活权”）。
        """
        if task_id is None:
            if self._focus_task_id:
                old = self._tasks.get(self._focus_task_id)
                if old and old.status == TaskStatus.FOCUS:
                    old.status = TaskStatus.READY
                    await self.save(old)
            self._focus_task_id = None
            return

        if self._focus_task_id and self._focus_task_id != task_id:
            old = self._tasks.get(self._focus_task_id)
            if old and old.status == TaskStatus.FOCUS:
                old.status = TaskStatus.SUSPENDED if preempt else TaskStatus.READY
                await self.save(old)
                if old.status == TaskStatus.SUSPENDED:
                    self._push_suspended(old.task_id)

        self._focus_task_id = task_id

        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.FOCUS
            await self.save(task)

    def current_focus(self) -> Optional[str]:
        """返回当前焦点任务 ID。"""
        return self._focus_task_id

    async def get_focus_task(self) -> Optional[Task]:
        """返回当前焦点任务对象（若存在）。"""
        if not self._focus_task_id:
            return None
        return await self.get(self._focus_task_id)

    # ---------- Internal helpers ----------
    def _push_suspended(self, task_id: str) -> None:
        """记录进入 SUSPENDED 的任务，维持 FIFO 顺序以便后续恢复。"""
        if task_id not in self._suspended_stack:
            self._suspended_stack.append(task_id)

    def _remove_suspended(self, task_id: str) -> None:
        if task_id in self._suspended_stack:
            self._suspended_stack = [tid for tid in self._suspended_stack if tid != task_id]
