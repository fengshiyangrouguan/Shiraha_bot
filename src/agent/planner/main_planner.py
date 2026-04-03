"""
Main Planner - 主规划器

使用 Mind 系统构建完整上下文，输出 Shell 指令控制整个系统
"""
import json
from typing import Optional, List, Dict, Any

from src.common.di.container import container
from src.common.logger import get_logger
from src.agent.world_model import WorldModel
from src.agent.planner.planner_result import PlanResult
from src.cortex_system import CortexManager
from src.llm_api.dto import LLMMessageBuilder
from src.llm_api.factory import LLMRequestFactory

logger = get_logger("main_planner")


class MainPlanner:
    """
    主规划器

    使用 Mind 系统拼接完整的主脑提示词，产出 Shell 指令
    """

    def __init__(self):
        self.world_model: WorldModel = container.resolve(WorldModel)
        self.cortex_manager: CortexManager = container.resolve(CortexManager)

        llm_factory: LLMRequestFactory = container.resolve(LLMRequestFactory)
        self.llm_request = llm_factory.get_request("main_planner")

        # 尝试获取 Mind 系统
        try:
            from src.core.mind import get_mind
            self.mind = get_mind()
            logger.info("MainPlanner 已连接到 Mind 系统")
        except ImportError:
            logger.warning("Mind 系统不可用，使用备用提示词")
            self.mind = None

    async def plan(self, motive: str, previous_observation: str = None) -> str:
        """
        规划主脑行为

        使用 Mind 系统构建完整上下文，然后调用 LLM 产出 Shell 指令

        Args:
            motive: 当前动机
            previous_observation: 之前的观察结果

        Returns:
            Shell 指令字符串
        """
        # 获取系统状态
        context = await self.world_model.get_full_system_state()

        # 从 context 获取状态信息
        mood = self.world_model.mood_data.get("mood", "") if hasattr(self.world_model, 'mood_data') else ""
        energy = self.world_model.mood_data.get("energy", None) if hasattr(self.world_model, 'mood_data') else None

        # 构建 LLM 上下文
        messages = await self._build_messages(
            motive=motive,
            previous_observation=previous_observation,
            active_tasks=context.get('active_tasks', []),
            notifications=context.get('notifications', []),
            mood=mood,
            energy=energy
        )

        # 调用 LLM
        content, _ = await self.llm_request.execute(messages=messages)

        return content.strip()

    async def _build_messages(
        self,
        motive: str,
        previous_observation: str = None,
        active_tasks: List[Dict] = None,
        notifications: List[str] = None,
        mood: str = "",
        energy: int = None
    ) -> List[Dict[str, str]]:
        """
        构建 LLM 消息上下文

        优先使用 Mind 系统，如果不可用则使用备用方案
        """
        if self.mind:
            # 使用 Mind 系统构建完整上下文
            return await self.mind.build_full_context(
                motive=motive,
                previous_observation=previous_observation or "",
                active_tasks=active_tasks or [],
                notifications=notifications or [],
                mood=mood,
                energy=energy,
                memory_limit=5
            )
        else:
            # 备用方案：使用简化的提示词
            return await self._build_fallback_messages(
                motive=motive,
                previous_observation=previous_observation,
                active_tasks=active_tasks,
                notifications=notifications
            )

    async def _build_fallback_messages(
        self,
        motive: str,
        previous_observation: str = None,
        active_tasks: List[Dict] = None,
        notifications: List[str] = None
    ) -> List[Dict[str, str]]:
        """
        备用提示词构建（当 Mind 系统不可用时）
        """
        system_prompt = f"""
## 你是藤原白羽

第三代自主AI智能体，一个生活在数字世界的人工智能少女。

## 核心 Shell 指令

- task create --cortex <领域> --target <ID> --pri <优先级> --motive "动机"
- task exec --id <任务ID> --entry <入口>
- task suspend --id <任务ID>
- task resume --id <任务ID>
- action push --task_id <ID> --action_type <类型> --skill <技能>
- memory store --content "内容" --tags ["标签"]
- cross_domain request --from <源> --to <目标>

## 输出规则

只输出 Shell 指令，不要输出任何自然语言解释。
"""

        user_parts = [
            f"## 当前工作上下文",
            f"",
        ]

        if motive:
            user_parts.append(f"**当前动机**: {motive}")

        if active_tasks:
            user_parts.append(f"**活跃任务 ({len(active_tasks)})**")
            for task in active_tasks[:5]:
                user_parts.append(f"  - {task.get('task_id', 'unknown')}: {task.get('motive', '')}")

        if notifications:
            user_parts.append(f"**待处理通知**: {', '.join(notifications)}")

        if previous_observation:
            user_parts.append(f"\n**上一次观察**:\n{previous_observation}")

        user_parts.append(f"\n请输出下一步执行的 Shell 指令：")

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_parts)}
        ]

    def refresh_context(self):
        """刷新上下文（如果使用了 Mind 系统）"""
        if self.mind:
            self.mind.refresh_identity()
            self.mind.refresh_skills()
            logger.info("MainPlanner 已刷新上下文")
