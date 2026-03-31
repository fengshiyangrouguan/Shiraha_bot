import asyncio
from typing import Dict, Any, Optional, List
from collections import deque

from src.common.logger import get_logger
from src.common.di.container import container
# from src.core.task.task_store import TaskStore

logger = get_logger("kernel_scheduler")

class Scheduler:
    """
    内核调度器 (Kernel Scheduler)
    负责管理就绪队列 (Ready Queue) 并执行抢占式调度逻辑。
    """

    def __init__(self):
        # 维护一个优先级队列，Key 为优先级 (0-100)，Value 为任务 ID 列表
        self.ready_queue: Dict[int, deque] = {}
        self.current_task_id: Optional[str] = None
        
        # 依赖注入：任务仓库，用于持久化任务状态
        # self.task_store: TaskStore = container.resolve(TaskStore)

    async def schedule(self) -> Optional[str]:
        """
        核心调度算法：找到当前优先级最高的任务 ID。
        """
        if not self.ready_queue:
            return None

        # 按优先级从高到低排序 (100 -> 0)
        sorted_priorities = sorted(self.ready_queue.keys(), reverse=True)
        
        for pri in sorted_priorities:
            if self.ready_queue[pri]:
                # 弹出最高优先级队列中的第一个任务
                next_task_id = self.ready_queue[pri].popleft()
                
                # 如果该优先级队列空了，清理键值
                if not self.ready_queue[pri]:
                    del self.ready_queue[pri]
                
                return next_task_id
        
        return None

    async def switch_context(self, next_task_id: str):
        """
        执行上下文切换 (Context Switch)
        """
        if self.current_task_id == next_task_id:
            return

        logger.info(f"🔄 Context Switch: {self.current_task_id} -> {next_task_id}")
        
        # 1. 挂起当前任务 (如果存在)
        if self.current_task_id:
            # await self.task_store.update_status(self.current_task_id, "SUSPENDED")
            pass

        # 2. 激活新任务
        self.current_task_id = next_task_id
        # await self.task_store.update_status(next_task_id, "RUNNING")

    async def add_to_ready(self, task_id: str, priority: int):
        """
        将任务加入就绪队列 (Ready Queue)
        """
        if priority not in self.ready_queue:
            self.ready_queue[priority] = deque()
        
        self.ready_queue[priority].append(task_id)
        logger.debug(f"Task {task_id} added to Ready Queue with priority {priority}")

    async def dispatch_to_executor(self, task_id: str, entry: str):
        """
        分发执行：真正调用子规划器的执行入口。
        这里模拟了 CPU 的指令周期。
        """
        await self.switch_context(task_id)
        
        # 这里的逻辑应该是：
        # 1. 获取该任务对应的 Sub-Planner 实例
        # 2. 调用其 run_step(entry)
        # 3. 处理返回的 Signal (FINISH, CALL_HELP 等)
        
        logger.info(f"🚀 Dispatching Task {task_id} to Entry: {entry}")
        # result = await sub_planner.run_step(entry)
        # return result

    async def handle_interrupt(self, interrupt_task_id: str, priority: int):
        """
        抢占式中断处理。
        如果新任务优先级更高，则强行中断当前任务。
        """
        current_pri = 0 # 假设从 task_store 获取当前任务优先级
        
        if priority > current_pri:
            logger.warning(f"⚠️ Preemptive Interrupt: New Task {interrupt_task_id}({priority}) > Current({current_pri})")
            # 将当前任务重新放回就绪队列
            if self.current_task_id:
                await self.add_to_ready(self.current_task_id, current_pri)
            
            # 立即切换
            await self.switch_context(interrupt_task_id)