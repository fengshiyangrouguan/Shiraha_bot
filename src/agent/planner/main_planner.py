"""
Main Planner - 主规划器

新版主规划器统一接收多种输入源，并输出结构化 JSON 规划结果。
"""
import json
from typing import Any, Dict, List, Optional

from src.agent.planner.planner_result import PlannerResult
from src.agent.world_model import WorldModel
from src.common.di.container import container
from src.common.logger import get_logger
from src.core.context.context_builder import ContextBuilder
from src.cortex_system import CortexManager
from src.llm_api.factory import LLMRequestFactory

logger = get_logger("main_planner")


class MainPlanner:
    """
    主规划器。

    职责：
    1. 收集当前世界状态与输入来源。
    2. 组装统一上下文。
    3. 调用 LLM，得到 `thought + shell_commands` 的结构化结果。
    """

    def __init__(self):
        self.world_model: WorldModel = container.resolve(WorldModel)
        self.cortex_manager: CortexManager = container.resolve(CortexManager)
        self.context_builder = ContextBuilder()

        llm_factory: LLMRequestFactory = container.resolve(LLMRequestFactory)
        self.llm_request = llm_factory.get_request("main_planner")

    async def plan(
        self,
        motive: str = "",
        previous_observation: Optional[str] = None,
        input_source: str = "idle_input",
        latest_signal: Optional[Dict[str, Any]] = None,
        debug_request: str = "",
    ) -> PlannerResult:
        """
        基于当前输入生成结构化规划结果。
        """
        system_state = await self.world_model.get_full_system_state()
        messages = await self._build_messages(
            motive=motive,
            previous_observation=previous_observation or system_state.get("last_observation", ""),
            active_tasks=system_state.get("active_tasks", []),
            notifications=self._normalize_notifications(system_state.get("notifications", {})),
            mood=getattr(self.world_model, "mood", ""),
            energy=getattr(self.world_model, "energy", None),
            input_source=input_source,
            latest_signal=latest_signal or {},
            debug_request=debug_request,
        )

        content, _ = await self.llm_request.execute(messages=messages)
        return self._parse_planner_output(content or "")

    async def _build_messages(
        self,
        motive: str,
        previous_observation: str = "",
        active_tasks: Optional[List[Dict[str, Any]]] = None,
        notifications: Optional[List[str]] = None,
        mood: str = "",
        energy: Optional[int] = None,
        input_source: str = "idle_input",
        latest_signal: Optional[Dict[str, Any]] = None,
        debug_request: str = "",
    ) -> List[Dict[str, str]]:
        """
        从 ContextBuilder 生成统一上下文。
        """
        return await self.context_builder.build_main_context(
            motive=motive,
            previous_observation=previous_observation,
            active_tasks=active_tasks or [],
            notifications=notifications or [],
            mood=mood,
            energy=energy,
            memory_limit=5,
            input_source=input_source,
            latest_signal=latest_signal or {},
            debug_request=debug_request,
        )

    def refresh_context(self):
        self.context_builder.refresh_identity()
        logger.info("MainPlanner 已刷新上下文")

    def _normalize_notifications(self, notifications: Any) -> List[str]:
        if isinstance(notifications, list):
            return [str(item) for item in notifications if item]

        if isinstance(notifications, dict):
            result = []
            for key, value in notifications.items():
                if value:
                    result.append(f"{key}: {value}")
            return result

        if notifications:
            return [str(notifications)]

        return []

    def _parse_planner_output(self, content: str) -> PlannerResult:
        """
        解析 Planner 输出。

        优先格式：
        ```json
        {"thought": "...", "shell_commands": ["..."]}
        ```

        回退策略：
        - 若是纯文本，则把每一行都当成 shell_commands。
        """
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                shell_commands = parsed.get("shell_commands", [])
                if isinstance(shell_commands, str):
                    shell_commands = [line.strip() for line in shell_commands.splitlines() if line.strip()]
                return PlannerResult(
                    thought=str(parsed.get("thought", "")).strip(),
                    shell_commands=[str(cmd).strip() for cmd in shell_commands if str(cmd).strip()],
                    raw_content=content,
                    metadata={
                        "format": "json",
                        "input_source": parsed.get("input_source", ""),
                    },
                )
        except Exception:
            pass

        fallback_commands = [line.strip() for line in cleaned.splitlines() if line.strip()]
        return PlannerResult(
            thought="",
            shell_commands=fallback_commands,
            raw_content=content,
            metadata={"format": "plain_text_fallback"},
        )
