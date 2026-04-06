from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.common.di.container import container
from src.common.logger import get_logger
from src.core.task.models import TaskMode
from src.core.task.task_manager import TaskManager
from src.cortex_system import CortexManager

from .reply_planner import ReplyPlanner

logger = get_logger("reply_runtime")


@dataclass
class ReplyRuntimeResult:
    handled: bool = False
    escalated: bool = False
    reason: str = ""
    task_id: str = ""
    reply_payload: Dict[str, Any] = field(default_factory=dict)


class ReplyRuntime:
    """
    回复器运行时。

    设计目标：
    1. 作为监听型任务的专用行为处理器。
    2. 面对高频社交消息时，优先自行消化，不频繁唤醒主 Planner。
    3. 只有无法处理时，才向主链升级为中断式规划。
    """

    def __init__(self):
        self.task_manager: TaskManager = container.resolve(TaskManager)
        self.cortex_manager: CortexManager = container.resolve(CortexManager)
        self.reply_planner = ReplyPlanner()

    async def handle_signal(self, signal) -> ReplyRuntimeResult:
        normalized_cortex = str(signal.source_cortex).lower()
        if normalized_cortex not in {"qq", "qq_chat", "qqchat"}:
            return ReplyRuntimeResult(handled=False, escalated=False, reason="非 QQ 信号")

        conversation_id = signal.target_id or "unknown_conversation"
        task = await self.task_manager.create_task(
            cortex="qq_chat",
            target_id=conversation_id,
            mode=TaskMode.LISTEN,
            priority=signal.priority,
            motive="监听并处理 QQ 会话消息",
            task_config={"listener_kind": "replyer", "conversation_id": conversation_id},
        )

        latest_signal = {
            "source_cortex": signal.source_cortex,
            "target_id": signal.target_id,
            "content": signal.content,
            "priority": getattr(signal.priority, "value", str(signal.priority)),
            "raw_data": signal.raw_data,
        }
        await self.task_manager.update_task_runtime(
            task.task_id,
            last_signal=latest_signal,
            append_window={
                "role": "observation",
                "content": f"[reply_signal] {signal.content}",
                "metadata": {"conversation_id": conversation_id},
            },
        )

        should_reply = self._quick_should_reply(signal)
        if not should_reply:
            return ReplyRuntimeResult(
                handled=True,
                escalated=False,
                reason="快速判断为不需要回应，保持沉默",
                task_id=task.task_id,
            )

        qq_cortex = self.cortex_manager.get_cortex("qq_chat")
        if qq_cortex is None:
            return ReplyRuntimeResult(
                handled=False,
                escalated=True,
                reason="QQ Cortex 不可用，升级给主 Planner",
                task_id=task.task_id,
            )

        try:
            conversation_context = await qq_cortex.get_conversation_context(conversation_id, limit=12)
            reply_plan = await self.reply_planner.plan_reply(
                conversation_id=conversation_id,
                latest_message=signal.raw_data.get("full_content", signal.content),
                conversation_context=conversation_context,
            )

            if not reply_plan.get("should_reply", False):
                return ReplyRuntimeResult(
                    handled=True,
                    escalated=False,
                    reason="高级回复规划器决定继续沉默",
                    task_id=task.task_id,
                    reply_payload=reply_plan,
                )

            reply_result = await qq_cortex.send_reply(
                conversation_id=conversation_id,
                content=str(reply_plan.get("reply_content", "")).strip(),
            )
            await self.task_manager.update_task_runtime(
                task.task_id,
                last_result={"reply_result": reply_result, "reply_plan": reply_plan},
                append_window={
                    "role": "assistant",
                    "content": str(reply_plan.get("reply_content", "")).strip(),
                    "metadata": {"style": reply_plan.get("style", "brief")},
                },
                increment_execution=True,
            )
            return ReplyRuntimeResult(
                handled=True,
                escalated=False,
                reason="回复器已完成回复",
                task_id=task.task_id,
                reply_payload={"reply_plan": reply_plan, "reply_result": reply_result},
            )
        except Exception as exc:
            logger.error(f"回复器处理失败，升级给主 Planner: {exc}", exc_info=True)
            return ReplyRuntimeResult(
                handled=False,
                escalated=True,
                reason=f"回复器无法自行解决: {exc}",
                task_id=task.task_id,
            )

    def _quick_should_reply(self, signal) -> bool:
        """
        第一层快速判断。

        这里尽量不调用大模型，只基于局部启发式决定是否值得继续深入处理。
        """
        raw_data = signal.raw_data or {}
        tags = set(raw_data.get("tags", []) or [])
        full_content = str(raw_data.get("full_content", signal.content or "")).strip()
        lowered = full_content.lower()

        if "at_me" in tags or "mentioned_me" in tags:
            return True
        if "?" in full_content or "？" in full_content:
            return True
        if any(keyword in lowered for keyword in ["白羽", "shiraha", "你在吗", "帮我", "请问"]):
            return True
        if len(full_content) <= 1:
            return False
        return False
