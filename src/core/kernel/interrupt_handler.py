import asyncio
from typing import Dict, Any, Optional
from src.common.logger import get_logger
from src.common.di.container import container

# 假设已有的内核组件
# from src.core.kernel.scheduler import Scheduler
# from src.agent.planners.main_planner import MainPlanner

logger = get_logger("kernel_interrupt")

class InterruptHandler:
    """
    内核中断处理器 (Interrupt Handler)
    负责接收来自 Cortex 的异步信号，并判断是否需要触发内核重调度。
    """

    def __init__(self):
        # self.scheduler: Scheduler = container.resolve(Scheduler)
        # self.main_planner: MainPlanner = container.resolve(MainPlanner)
        
        # 中断屏蔽字 (Interrupt Mask)：用于在某些高优任务执行时，暂时忽略低优中断
        self.interrupt_mask_level = 0 

    async def handle_external_event(self, source_cortex: str, raw_data: Dict[str, Any]):
        """
        外部事件入口 (Entry Point for Cortex)
        source_cortex: 来源驱动名称 (如 'qq_cortex')
        raw_data: 原始负载 (消息内容、发送者等)
        """
        logger.info(f"📥 Received External Interrupt from {source_cortex}")

        # 1. 预处理 (Pre-processing)
        # 提取关键元数据，如紧急程度、来源 ID 等
        event_metadata = self._extract_metadata(raw_data)
        
        # 2. 中断过滤 (Filtering)
        if event_metadata['priority'] < self.interrupt_mask_level:
            logger.debug(f"🔇 Interrupt masked: {event_metadata['priority']} < {self.interrupt_mask_level}")
            # 可以选择存入待处理缓冲区 (SoftIRQ)
            return

        # 3. 触发内核决策 (Trigger Kernel Decision)
        # 就像 CPU 触发中断向量表，我们需要唤醒 Main Planner
        await self._invoke_reschedule(source_cortex, event_metadata)

    def _extract_metadata(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        静态启发式评估 (Heuristic Evaluation)
        在不消耗大量 LLM Token 的情况下，对事件进行初步画像。
        """
        # 示例逻辑：如果是特定的关键词或特定的人，初始优先级设高
        content = raw_data.get("message", "")
        priority = 50 # 默认优先级
        
        if "@me" in content or "紧急" in content:
            priority = 90
            
        return {
            "target_id": raw_data.get("user_id") or raw_data.get("group_id"),
            "content": content,
            "priority": priority,
            "raw": raw_data
        }

    async def _invoke_reschedule(self, source: str, metadata: Dict[str, Any]):
        """
        唤醒主规划器进行全局重调度
        """
        logger.warning(f"🔔 Raising Interruption Request (IRQ) for {source}")

        # 将当前中断信息作为 "previous_observation" 喂给 Main Planner
        observation = (
            f"INTERRUPT_SIGNAL: New message from {source}. "
            f"Target: {metadata['target_id']}. Content: {metadata['content'][:20]}..."
        )

        # 核心逻辑：
        # 1. 运行 Main Planner 获取新的 Shell 指令集
        # 2. 将指令集交给 Interpreter 执行
        # 3. 如果指令集中包含 'task suspend'，Scheduler 会执行上下文切换
        
        # commands = await self.main_planner.plan(motive="Handle Interrupt", previous_observation=observation)
        # await container.resolve('Interpreter').execute_batch(commands)

    def set_mask_level(self, level: int):
        """
        设置中断屏蔽等级。比如在执行“深度思考”任务时，屏蔽所有优先级低于 80 的中断。
        """
        self.interrupt_mask_level = level
        logger.info(f"🛡️ Kernel Interrupt Mask Level set to {level}")