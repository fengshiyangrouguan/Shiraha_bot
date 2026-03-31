import time
from typing import Dict, Any, Optional
from src.common.logger import get_logger

logger = get_logger("ipc_shm")

class SharedMemory:
    """
    模拟 Linux 共享内存 (Shared Memory)
    用于存放跨任务的上下文注入数据。
    """
    def __init__(self):
        # 结构: { "context_ref": { "data": Any, "timestamp": float, "owner_task": str } }
        self._memory: Dict[str, Dict[str, Any]] = {}

    async def write(self, context_ref: str, key: str, value: Any, task_id: str):
        """
        向指定的上下文引用地址写入数据
        """
        if context_ref not in self._memory:
            self._memory[context_ref] = {"_meta": {"created_at": time.time()}}
        
        self._memory[context_ref][key] = value
        self._memory[context_ref]["_meta"]["last_writer"] = task_id
        logger.info(f"📝 SHM Write: [{context_ref}] -> Key: {key} (by {task_id})")

    async def read(self, context_ref: str, key: Optional[str] = None) -> Any:
        """
        从指定的上下文引用地址读取数据
        """
        ctx = self._memory.get(context_ref)
        if not ctx:
            return None
        
        if key:
            return ctx.get(key)
        return ctx

    async def clear(self, context_ref: str):
        """清理内存地址 (类似于 shmrm)"""
        if context_ref in self._memory:
            del self._memory[context_ref]
            logger.debug(f"🧹 SHM Cleared: {context_ref}")

    def list_segments(self):
        """查看当前所有内存分段状态"""
        return list(self._memory.keys())