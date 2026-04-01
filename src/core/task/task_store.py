from typing import Dict, Optional, List
from src.core.task.models import TaskInstance, TaskStatus
from src.common.logger import get_logger

logger = get_logger("task_store")

class TaskStore:
    """
    任务状态仓库
    目前为内存存储，后续可轻松扩展为 Redis 或 SQLite 持久化
    """
    def __init__(self):
        self._tasks: Dict[str, TaskInstance] = {}

    async def save(self, task: TaskInstance):
        # 更新时间戳，后续可替换为数据库的 update_at 逻辑
        import time
        task.updated_at = time.time()
        self._tasks[task.task_id] = task
        return task

    async def get(self, task_id: str) -> Optional[TaskInstance]:
        return self._tasks.get(task_id)

    async def get_by_target(self, target_id: str) -> List[TaskInstance]:
        """按目标ID查找任务（如：检查某个群是否已有活跃任务）"""
        return [t for t in self._tasks.values() if t.target_id == target_id and t.status != TaskStatus.TERMINATED]

    async def list_all(self, status: Optional[TaskStatus] = None) -> List[TaskInstance]:
        if status:
            return [t for t in self._tasks.values() if t.status == status]
        return list(self._tasks.values())

    async def delete(self, task_id: str):
        if task_id in self._tasks:
            del self._tasks[task_id]

    async def update_status(self, task_id: str, status: TaskStatus):
        task = await self.get(task_id)
        if not task:
            return None
        task.status = status
        await self.save(task)
        return task
