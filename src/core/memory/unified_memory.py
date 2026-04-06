"""
Unified Memory - 统一记忆系统

整合工作记忆、短期记忆和长期记忆，提供统一的记忆 API。
支持跨域、跨时间的记忆检索和引用。
"""
import time
from typing import List, Optional, Dict, Any, Tuple
import asyncio
from collections import deque

from .memory_entry import MemoryEntry, MemoryType
from .working_memory import WorkingMemory
from .long_term_memory import LongTermMemory
from .memory_retriever import MemoryRetriever
from src.common.logger import get_logger

logger = get_logger("unified_memory")


class UnifiedMemory:
    """
    统一记忆系统

    整合多层次记忆，提供统一的存储和检索接口
    """

    def __init__(
        self,
        working_capacity: int = 20,
        stm_capacity: int = 100,
        stm_ttl_hours: float = 24
    ):
        # 工作记忆：小容量，快速访问
        self.working = WorkingMemory(max_capacity=working_capacity)

        # 短期记忆：中等容量，相对快速
        self.short_term: deque = deque(maxlen=stm_capacity)
        self.stm_ttl = stm_ttl_hours * 3600  # TTL 秒数

        # 长期记忆：数据库 + 向量存储
        self.long_term: Optional[LongTermMemory] = None

        # 检索器：语义检索
        self.retriever: Optional[MemoryRetriever] = None

        self._initialized = False

    async def initialize(self, long_term_memory: LongTermMemory, retriever: MemoryRetriever):
        """初始化长期记忆和检索器"""
        self.long_term = long_term_memory
        self.retriever = retriever
        self._initialized = True
        logger.info("UnifiedMemory 初始化完成")

    async def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        source_cortex: str = "",
        source_target: str = "",
        source_action: Optional[str] = None,
        tags: List[str] = None,
        keywords: List[str] = None,
        importance: float = 0.5,
        task_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        存储记忆

        Args:
            content: 记忆内容
            memory_type: 记忆类型
            source_cortex: 来源域
            source_target: 来源目标
            source_action: 来源行为
            tags: 标签
            keywords: 关键词
            importance: 重要性（0-1）
            task_id: 关联任务ID
            embedding: 预计算的向量（可选）

        Returns:
            memory_id
        """
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            source_cortex=source_cortex,
            source_target=source_target,
            source_action=source_action,
            tags=tags or [],
            keywords=keywords or [],
            importance=importance,
            related_task_id=task_id,
            embedding=embedding,
        )

        if memory_type == MemoryType.WORKING:
            # 存储到工作记忆
            memory_id = self.working.store(entry, task_id=task_id)

            # 同时存储到短期记忆，作为备份
            self.short_term.append(entry)

        elif memory_type == MemoryType.SHORT_TERM:
            # 存储到短期记忆
            self.short_term.append(entry)

            # 生成向量
            if self.retriever and not entry.embedding:
                entry.embedding = await self.retriever.embed(content)

        elif memory_type == MemoryType.LONG_TERM:
            # 存储到长期记忆
            if self.long_term:
                if self.retriever and not entry.embedding:
                    entry.embedding = await self.retriever.embed(content)
                memory_id = await self.long_term.store(entry)
            else:
                logger.warning("长期记忆未初始化，降级存储到短期记忆")
                self.short_term.append(entry)
                memory_id = entry.memory_id

        logger.debug(f"存储记忆: {memory_id} type={memory_type.value} content={content[:30]}...")
        return memory_id

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        memory_types: List[MemoryType] = None,
        source_cortex: str = "",
        source_target: str = "",
        task_id: str = "",
        semantic: bool = True,
    ) -> List[MemoryEntry]:
        """
        检索记忆

        Args:
            query: 查询内容或关键词
            limit: 返回数量限制
            memory_types: 记忆类型过滤
            source_cortex: 来源域过滤
            source_target: 来源目标过滤
            task_id: 任务ID过滤
            semantic: 是否使用语义检索

        Returns:
            记忆条目列表
        """
        results = []

        # 指定记忆类型
        types = memory_types or [MemoryType.WORKING, MemoryType.SHORT_TERM, MemoryType.LONG_TERM]

        # 1. 从工作记忆检索
        if MemoryType.WORKING in types:
            if task_id:
                results.extend(self.working.retrieve_by_task(task_id, limit=limit))
            elif source_target:
                results.extend(self.working.retrieve_by_target(source_target, limit=limit))
            else:
                results.extend(self.working.search_content(query, limit=limit))

        # 2. 从短期记忆检索
        if MemoryType.SHORT_TERM in types and len(results) < limit:
            for entry in list(self.short_term):
                # 过滤条件
                if source_cortex and entry.source_cortex != source_cortex:
                    continue
                if source_target and entry.source_target != source_target:
                    if query.lower() not in entry.content.lower():
                        continue
                if task_id and entry.related_task_id != task_id:
                    continue

                # 简单匹配
                if query.lower() in entry.content.lower():
                    entry.mark_accessed()
                    results.append(entry)
                    if len(results) >= limit:
                        break

        # 3. 从长期记忆检索
        if MemoryType.LONG_TERM in types and self.long_term and len(results) < limit:
            limit_ltm = limit - len(results)

            if semantic and self.retriever:
                # 语义检索
                ltm_results = await self.retriever.semantic_search(
                    query,
                    limit=limit_ltm,
                    source_cortex=source_cortex,
                    source_target=source_target
                )
                results.extend(ltm_results)
            else:
                # 精确检索
                ltm_results = await self.long_term.retrieve(query, limit=limit_ltm)
                results.extend(ltm_results)

        # 去重
        seen_ids = set()
        unique_results = []
        for entry in results:
            if entry.memory_id not in seen_ids:
                seen_ids.add(entry.memory_id)
                unique_results.append(entry)

        return unique_results[:limit]

    async def forget(self, memory_id: str) -> bool:
        """
        遗忘记忆

        从所有记忆层次中删除指定记忆
        """
        # 工作记忆
        if self.working.forget(memory_id):
            logger.debug(f"从工作记忆遗忘: {memory_id}")
            return True

        # 短期记忆
        for i, entry in enumerate(self.short_term):
            if entry.memory_id == memory_id:
                self.short_term.remove(entry)
                logger.debug(f"从短期记忆遗忘: {memory_id}")
                return True

        # 长期记忆
        if self.long_term:
            success = await self.long_term.delete(memory_id)
            if success:
                logger.debug(f"从长期记忆遗忘: {memory_id}")
                return True

        logger.warning(f"未找到记忆: {memory_id}")
        return False

    async def clear_task_memories(self, task_id: str):
        """清除特定任务的所有记忆"""
        self.working.clear_task_memories(task_id)

        # 短期记忆也需要清理
        to_remove = [entry for entry in self.short_term if entry.related_task_id == task_id]
        for entry in to_remove:
            self.short_term.remove(entry)

        logger.debug(f"清除任务记忆: {task_id}")

    async def get_context_messages(
        self,
        task_id: str = "",
        source_cortex: str = "",
        source_target: str = "",
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        获取上下文消息列表（统一 {role, content} 格式）

        用于构建 LLM 调用的上下文
        """
        memories = await self.retrieve(
            query="",
            task_id=task_id,
            source_cortex=source_cortex,
            source_target=source_target,
            limit=limit,
            semantic=False
        )

        messages = []
        for entry in memories:
            # 根据记忆类型转换为消息格式
            if entry.memory_type == MemoryType.WORKING:
                role = "assistant"  # 工作记忆通常是自己的思考
            elif entry.memory_type == MemoryType.SHORT_TERM:
                role = "memory"
            else:
                role = "memory"

            # 添加来源信息
            source_info = ""
            if entry.source_cortex:
                source_info = f"[{entry.source_cortex}"
                if entry.source_target:
                    source_info += f":{entry.source_target}"
                source_info += "] "

            messages.append({
                "role": role,
                "content": f"{source_info}{entry.content}"
            })

            # 如果有相关任务ID，也添加进去
            if entry.related_task_id:
                messages[-1]["task_id"] = entry.related_task_id

        return messages

    async def cleanup_expired(self):
        """清理过期的记忆"""
        # 短期记忆清理
        current_time = time.time()
        initial_size = len(self.short_term)
        self.short_term = deque(
            [e for e in self.short_term if not e.is_expired(self.stm_ttl)],
            maxlen=self.short_term.maxlen
        )
        removed = initial_size - len(self.short_term)
        if removed > 0:
            logger.debug(f"清理过期短期记忆: {removed} 条")

        # 长期记忆也需要定期清理
        if self.long_term:
            await self.long_term.cleanup()

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "working_count": self.working.size(),
            "short_term_count": len(self.short_term),
            "long_term_count": self.long_term.size() if self.long_term else 0,
            "initialized": self._initialized
        }
