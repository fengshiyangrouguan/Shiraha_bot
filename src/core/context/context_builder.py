"""
Context Builder - 上下文构建器

从各个模块收集信息，构建统一上下文。
"""
from typing import List, Dict, Any, Optional
from .unified_context import UnifiedContext, ContextMessage, Role
from src.core.memory import UnifiedMemory, MemoryEntry
from src.common.logger import get_logger

logger = get_logger("context_builder")


class ContextBuilder:
    """
    上下文构建器

    从多个来源收集信息，构建供 LLM 使用的统一上下文
    """

    def __init__(self, unified_memory: UnifiedMemory):
        self.memory = unified_memory

    async def build_context(
        self,
        task_id: str = "",
        source_cortex: str = "",
        source_target: str = "",
        max_tokens: int = 4096,
        include_system: bool = True,
        include_memory: bool = True,
        history_limit: int = 10
    ) -> UnifiedContext:
        """
        构建完整上下文

        Args:
            task_id: 关联任务ID
            source_cortex: 来源域
            source_target: 来源目标
            max_tokens: 最大 token 数
            include_system: 是否包含系统消息
            include_memory: 是否包含记忆
            history_limit: 历史消息数量限制

        Returns:
            UnifiedContext 实例
        """
        context = UnifiedContext()
        context._max_tokens = max_tokens

        # 1. 添加系统消息
        if include_system:
            await self._add_system_messages(context, source_cortex)

        # 2. 添加记忆消息
        if include_memory:
            await self._add_memory_messages(context, task_id, source_cortex, source_target)

        # 3. 添加历史对话（从记忆中）
        await self._add_history_messages(context, task_id, source_cortex, source_target, history_limit)

        # 4. 截断以适应 token 限制
        context.truncate(max_tokens)

        return context

    async def _add_system_messages(self, context: UnifiedContext, cortex: str = ""):
        """添加系统消息"""
        # TODO: 从配置加载系统消息
        # 这里可以加载 persona、skill 等系统消息

        # 默认系统消息
        default_system = """你是 Shiraha，一个有着人类行为的智能助手。

你的特点：
- 你有单注意力头，一次只能专注于一件事
- 你可以同时在意多个任务，但只能执行其中一个
- 你的行为会根据所处的环境和互动而演化
- 你可以跨域思考，在不同场景间建立联系"""
        context.append({
            "role": Role.SYSTEM.value,
            "content": default_system,
            "metadata": {"source": "system_default"}
        })

        # 如果指定了 cortex，添加 cortex 特定的系统消息
        if cortex:
            # TODO: 从 cortex 配置或 skill 文件加载
            cortex_system = f"你正在 {cortex} 域中工作。"
            context.append({
                "role": Role.SYSTEM.value,
                "content": cortex_system,
                "metadata": {"source": cortex}
            })

    async def _add_memory_messages(
        self,
        context: UnifiedContext,
        task_id: str,
        source_cortex: str,
        source_target: str
    ):
        """添加记忆消息"""
        # 从统一记忆系统检索相关记忆
        memories = await self.memory.retrieve(
            query="",
            task_id=task_id,
            source_cortex=source_cortex,
            source_target=source_target,
            limit=5,
            semantic=False
        )

        for memory in memories:
            context.append({
                "role": Role.MEMORY.value,
                "content": memory.content,
                "metadata": {
                    "source_cortex": memory.source_cortex,
                    "source_target": memory.source_target,
                    "importance": memory.importance,
                    "timestamp": memory.timestamp
                },
                "timestamp": memory.timestamp,
                "task_id": memory.related_task_id
            })

    async def _add_history_messages(
        self,
        context: UnifiedContext,
        task_id: str,
        source_cortex: str,
        source_target: str,
        limit: int
    ):
        """添加历史对话消息"""
        # TODO: 从对话历史中获取消息
        # 这里可以合并系统的聊天历史

        # 临时：从记忆中查找历史
        memories = await self.memory.retrieve(
            query="chat conversation history",
            task_id=task_id,
            source_cortex=source_cortex,
            source_target=source_target,
            limit=limit,
            semantic=False
        )

        for memory in memories:
            # 判断消息类型
            if "user:" in memory.content[:10].lower():
                role = Role.USER.value
                content = memory.content.replace("user:", "", 1).strip()
            elif memory.source_action and "reply" in memory.source_action.lower():
                role = Role.ASSISTANT.value
                content = memory.content
            else:
                role = Role.MEMORY.value
                content = memory.content

            context.append({
                "role": role,
                "content": content,
                "metadata": {
                    "source_cortex": memory.source_cortex,
                    "timestamp": memory.timestamp
                },
                "timestamp": memory.timestamp
            })

    async def build_planner_context(
        self,
        current_motive: str,
        focus_task_id: str,
        world_state: Dict[str, Any]
    ) -> UnifiedContext:
        """
        构建 Planner 专用上下文

        Args:
            current_motive: 当前动机
            focus_task_id: 焦点任务ID
            world_state: 世界状态

        Returns:
            Planner 使用的上下文
        """
        context = UnifiedContext()

        # 系统消息：Planner 角色
        context.append({
            "role": Role.SYSTEM.value,
            "content": """你是全局任务调度规划器。

你的职责：
1. 基于当前动机和世界状态，决定下一步执行什么
2. 保持单注意力头，同时只能有一个任务获得焦点
3. 使用 Shell 指令控制整个系统
4. 产出格式：shell_commands（每行一个指令）

可用的主要指令：
- task create/exec/suspend/block/kill
- action push/pop/complete
- memory store/retrieve
- context load/append
- signal emit/broadcast

注意：
- 使用类 Bash 的 Shell 指令格式
- 指令应该简洁，具体实现由执行层处理"""
        })

        # 当前动机
        if current_motive:
            context.append({
                "role": Role.OBSERVATION.value,
                "content": f"当前动机：{current_motive}"
            })

        # 焦点任务信息
        if focus_task_id:
            context.append({
                "role": Role.OBSERVATION.value,
                "content": f"焦点任务：{focus_task_id}"
            })

        # 世界状态
        if world_state:
            state_summary = self._format_world_state(world_state)
            context.append({
                "role": Role.OBSERVATION.value,
                "content": f"世界状态：\n{state_summary}"
            })

        return context

    def _format_world_state(self, state: Dict[str, Any]) -> str:
        """格式化世界状态"""
        parts = []
        for key, value in state.items():
            if isinstance(value, (list, dict)):
                parts.append(f"{key}: {str(value)[:100]}...")
            else:
                parts.append(f"{key}: {value}")
        return "\n".join(parts)

    async def build_evaluator_context(
        self,
        task_context: str,
        metrics: List[str]
    ) -> UnifiedContext:
        """
        构建评估器专用上下文（用于小模型状态评估）

        Args:
            task_context: 任务上下文
            metrics: 需要评估的指标列表

        Returns:
            评估器使用的上下文
        """
        context = UnifiedContext()

        context.append({
            "role": Role.SYSTEM.value,
            "content": """你是状态评估器。

你的职责：
1. 基于提供的上下文，评估指定的状态指标
2. 输出格式：JSON，包含每个指标的数值

指标说明：
- stress (0-1): 压力值，0=轻松，1=高压
- focus (0-1): 专注度，0=分散，1=专注
- energy (0-1): 能量值，0=疲惫，1=充满活力
- interest (0-1): 兴趣度，0=无趣，1=非常感兴趣"""
        })

        context.append({
            "role": Role.USER.value,
            "content": f"""任务上下文：
{task_context}

需要评估的指标：
{", ".join(metrics)}

请以 JSON 格式输出评估结果。"""
        })

        return context

    def create_signal_message(
        self,
        signal_type: str,
        content: str,
        source_cortex: str,
        source_target: str = ""
    ) -> ContextMessage:
        """
        创建信号消息

        Args:
            signal_type: 信号类型
            content: 信号内容
            source_cortex: 来源域
            source_target: 来源目标

        Returns:
            ContextMessage 实例
        """
        return ContextMessage(
            role=Role.OBSERVATION.value,
            content=f"[{signal_type}] {content}",
            metadata={
                "signal_type": signal_type,
                "source": f"{source_cortex}:{source_target}" if source_target else source_cortex
            },
            source_cortex=source_cortex,
            source_target=source_target
        )
