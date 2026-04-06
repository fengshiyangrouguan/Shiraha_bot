"""
Working Memory - 工作记忆

工作记忆是小容量的内存存储，用于存储当前正在关注的信息。
特点是：快速访问、容量有限、生命周期短。
"""
import time
from typing import List, Optional, Dict, Any, Set
from collections import deque
from .memory_entry import MemoryEntry, MemoryType


class WorkingMemory:
    """
    工作记忆

    容量有限，用于存储当前任务上下文和临时信息
    """

    def __init__(self, max_capacity: int = 20):
        self.max_capacity = max_capacity
        self._memory: deque = deque(maxlen=max_capacity)
        self._by_task: Dict[str, Set[str]] = {}  # task_id -> memory_ids
        self._by_target: Dict[str, Set[str]] = {}  # target_id -> memory_ids

    def store(self, entry: MemoryEntry, task_id: Optional[str] = None) -> str:
        """
        存储记忆条目到工作记忆

        Args:
            entry: 记忆条目
            task_id: 关联的任务ID

        Returns:
            memory_id
        """
        if len(self._memory) >= self.max_capacity:
            # 容量满时，移除最旧且访问次数最少的
            self._evict()

        entry.memory_type = MemoryType.WORKING
        self._memory.append(entry)

        # 建立索引
        if entry.memory_id not in self._by_target:
            self._by_target[entry.memory_id] = set()
        self._by_target[entry.memory_id].add(entry.source_target)

        if task_id:
            if task_id not in self._by_task:
                self._by_task[task_id] = set()
            self._by_task[task_id].add(entry.memory_id)

        return entry.memory_id

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """获取指定的记忆条目"""
        for entry in self._memory:
            if entry.memory_id == memory_id:
                entry.mark_accessed()
                return entry
        return None

    def retrieve_by_task(self, task_id: str, limit: int = 5) -> List[MemoryEntry]:
        """按任务ID检索相关记忆"""
        if task_id not in self._by_task:
            return []

        memory_ids = list(self._by_task[task_id])[:limit]
        results = []
        for memory_id in memory_ids:
            entry = self.get(memory_id)
            if entry:
                results.append(entry)
        return results

    def retrieve_by_target(self, target_id: str, limit: int = 5) -> List[MemoryEntry]:
        """按目标ID检索相关记忆"""
        results = []
        for entry in self._memory:
            if entry.source_target == target_id:
                entry.mark_accessed()
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def retrieve_by_tags(self, tags: List[str], limit: int = 5) -> List[MemoryEntry]:
        """按标签检索相关记忆"""
        tag_set = set(tags)
        results = []
        for entry in self._memory:
            if tag_set.intersection(set(entry.tags)):
                entry.mark_accessed()
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def search_content(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """简单的内容搜索"""
        query_lower = query.lower()
        results = []
        for entry in self._memory:
            if query_lower in entry.content.lower():
                entry.mark_accessed()
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def clear_task_memories(self, task_id: str):
        """清除特定任务的所有工作记忆"""
        if task_id not in self._by_task:
            return

        memory_ids_to_remove = self._by_task[task_id].copy()
        for memory_id in memory_ids_to_remove:
            self.forget(memory_id)

        del self._by_task[task_id]

    def forget(self, memory_id: str) -> bool:
        """遗忘指定的记忆条目"""
        for i, entry in enumerate(self._memory):
            if entry.memory_id == memory_id:
                # 清理索引
                self._by_target.pop(memory_id, None)
                for task_ids in self._by_task.values():
                    task_ids.discard(memory_id)

                self._memory.remove(entry)
                return True
        return False

    def clear_all(self):
        """清除所有工作记忆"""
        self._memory.clear()
        self._by_task.clear()
        self._by_target.clear()

    def _evict(self):
        """驱逐最不重要的记忆"""
        if not self._memory:
            return

        # 按访问次数和重要性排序，驱逐最不重要的
        sorted_items = sorted(
            list(self._memory),
            key=lambda x: (x.access_count, x.importance, x.timestamp)
        )

        # 移除最旧的一个
        if sorted_items:
            self.forget(sorted_items[0].memory_id)

    def size(self) -> int:
        """返回当前记忆数量"""
        return len(self._memory)

    def get_all(self) -> List[MemoryEntry]:
        """获取所有工作记忆"""
        return list(self._memory)
