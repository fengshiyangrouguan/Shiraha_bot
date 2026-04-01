import asyncio
import json
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.agent.motive.motive_engine import MotiveEngine
from src.agent.planner.main_planner import MainPlanner
from src.agent.planner.planner_result import PlanResult
from src.agent.world_model import WorldModel
from src.common.action_model.action_spec import ActionSpec
from src.common.action_model.tool_result import ToolResult
from src.common.database.database_manager import DatabaseManager
from src.common.database.database_model import BehaviorHistoryDB
from src.common.di.container import container
from src.common.logger import get_logger
from src.cortices.manager import CortexManager
from src.llm_api.factory import LLMRequestFactory
from src.core.task.task_store import TaskStore as CoreTaskStore
from src.core.task.task_manager import TaskManager
from src.core.kernel.scheduler import Scheduler
from src.core.kernel.interpreter import KernelInterpreter
from src.core.kernel.interrupt_handler import InterruptHandler

logger = get_logger("agent_loop")

PipelineHook = Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]] | Optional[Dict[str, Any]]]


class AgentLoop:
    agent_interrupt_event = asyncio.Event()

    def __init__(self):
        logger.info("初始化 AgentLoop...")
        # ---- 注册核心内核组件（供 WorldModel/Planner 共享） ----
        self.core_task_store = CoreTaskStore()
        try:
            container.register_instance(CoreTaskStore, self.core_task_store)
        except Exception:
            # 已注册则忽略
            pass

        self.task_manager = TaskManager(self.core_task_store)
        self.scheduler = Scheduler()
        self.kernel_interpreter = KernelInterpreter()
        self.interrupt_handler = InterruptHandler()
        try:
            container.register_instance(TaskManager, self.task_manager)
            container.register_instance(Scheduler, self.scheduler)
            container.register_instance(KernelInterpreter, self.kernel_interpreter)
        except Exception:
            pass

        # ---- Agent 认知层组件 ----
        self.world_model: WorldModel = container.resolve(WorldModel)
        self.motive_engine = MotiveEngine()
        self.main_planner = MainPlanner()
        self.cortex_manager: CortexManager = container.resolve(CortexManager)
        self.llm_factory: LLMRequestFactory = container.resolve(LLMRequestFactory)
        self.database_manager: DatabaseManager = container.resolve(DatabaseManager)

        self._is_running = False
        self._main_task: Optional[asyncio.Task] = None
        self.heartbeat_interval = 5
        self._pipeline_hooks: Dict[str, List[PipelineHook]] = {}

    def register_pipeline_hook(self, stage: str, hook: PipelineHook) -> None:
        self._pipeline_hooks.setdefault(stage, []).append(hook)

    async def _emit_stage(self, stage: str, context: Dict[str, Any]) -> Dict[str, Any]:
        hooks = self._pipeline_hooks.get(stage, [])
        current_context = context

        for hook in hooks:
            result = hook(current_context)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, dict):
                current_context = result

        return current_context

    def _should_record_immediate_memory(self, action: ActionSpec, result: ToolResult) -> bool:
        if not action or not result:
            return False
        if action.action_type != "tool":
            return False
        if not result.success or not result.summary:
            return False
        return action.tool_name in {"get_current_weather", "quick_reply"}

    def _record_immediate_memory(self, action: ActionSpec, result: ToolResult) -> None:
        if not self._should_record_immediate_memory(action, result):
            return
        self.world_model.add_memory(result.summary.strip())

    def _guess_source_cortex(self, actions: List[ActionSpec]) -> Optional[str]:
        cortex_names: List[str] = []
        for action in actions:
            descriptor = self.cortex_manager.get_tool_descriptor(action.tool_name)
            if not descriptor:
                continue
            source = descriptor.source or ""
            if source.startswith("cortex:"):
                parts = source.split(".")
                if "cortices" in parts:
                    cortex_index = parts.index("cortices") + 1
                    if cortex_index < len(parts):
                        cortex_name = parts[cortex_index]
                        if cortex_name not in cortex_names:
                            cortex_names.append(cortex_name)
            elif source.startswith("plugin:") and "plugin" not in cortex_names:
                cortex_names.append("plugin")

        if not cortex_names:
            return None
        if len(cortex_names) == 1:
            return cortex_names[0]
        return "mixed"

    def _extract_conversation_id(self, actions: List[ActionSpec]) -> Optional[str]:
        for action in actions:
            conversation_id = (
                (action.parameters or {}).get("conversation_id")
                or (action.metadata or {}).get("conversation_id")
            )
            if conversation_id:
                return str(conversation_id)
        return None

    async def _extract_behavior_history_payload(
        self,
        motive: str,
        initial_plan: PlanResult,
        results: List[ToolResult],
        actions: List[ActionSpec],
        final_memory: str,
    ) -> Dict[str, Any]:
        action_names = [action.tool_name for action in actions]
        action_log = "\n".join(
            f"步骤 {index + 1}: action={action.tool_name}, summary={result.summary}"
            for index, (action, result) in enumerate(zip(actions, results))
        )

        prompt = f"""
请把下面这轮 agent 行为历史抽取为结构化 JSON，用于长期记忆中的“行为历史”表。

## 已知信息
- motive: {motive}
- initial_plan_reason: {initial_plan.reason}
- final_memory_summary: {final_memory}
- executed_actions: {json.dumps(action_names, ensure_ascii=False)}

## action_log
{action_log}

请输出 JSON，字段如下：
{{
  "scene": "一句简短场景名，例如 group_chat / private_chat / reading / planning",
  "memory_type": "behavior_history",
  "keywords": ["关键词1", "关键词2"],
  "tags": ["标签1", "标签2"],
  "importance": 0~1的浮点数值,
  "notes": "可选，一句补充说明"
}}

要求：
1. 只输出 JSON。
2. keywords 和 tags 最多各 6 个。
3. scene 用短词，不要写成长句。
4. memory_type 固定返回 behavior_history。
"""

        try:
            llm_request = self.llm_factory.get_request("utils_small")
            content, _ = await llm_request.execute(prompt=prompt)
            payload_text = (content or "").strip()
            if payload_text.startswith("```"):
                payload_text = payload_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(payload_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            logger.warning(f"行为历史抽取失败，使用兜底结构: {exc}")

        return {
            "scene": "general",
            "memory_type": "behavior_history",
            "keywords": action_names[:6],
            "tags": [self._guess_source_cortex(actions) or "agent"],
            "importance": "medium",
            "notes": "",
        }

    async def _persist_behavior_history(
        self,
        motive: str,
        initial_plan: PlanResult,
        results: List[ToolResult],
        actions: List[ActionSpec],
        final_memory: str,
    ) -> None:
        if not final_memory:
            return

        extracted = await self._extract_behavior_history_payload(
            motive=motive,
            initial_plan=initial_plan,
            results=results,
            actions=actions,
            final_memory=final_memory,
        )

        record = BehaviorHistoryDB(
            memory_id=str(uuid.uuid4()),
            created_at=time.time(),
            summary=final_memory,
            motive=motive,
            initial_plan_reason=initial_plan.reason,
            source_cortex=self._guess_source_cortex(actions),
            scene=str(extracted.get("scene") or "general"),
            memory_type=str(extracted.get("memory_type") or "behavior_history"),
            conversation_id=self._extract_conversation_id(actions),
            source_tools=[action.tool_name for action in actions] or None,
            keywords=[str(item) for item in (extracted.get("keywords") or [])][:6] or None,
            tags=[str(item) for item in (extracted.get("tags") or [])][:6] or None,
            importance = extracted.get("importance"))
        await self.database_manager.upsert(record)
        logger.info(f"已写入行为历史长期记忆: memory_id={record.memory_id}")

    async def _execute_action(
        self,
        action: ActionSpec,
        chain_step: int,
        motive: str,
        previous_results: List[ToolResult],
    ) -> ToolResult:
        before_context = await self._emit_stage(
            "before_action_execute",
            {
                "motive": motive,
                "chain_step": chain_step,
                "action": action,
                "previous_results": previous_results,
            },
        )
        action = before_context.get("action", action)

        try:
            result = await self.cortex_manager.execute_action(action)
        except Exception as exc:
            error_result = ToolResult(
                success=False,
                summary=f"执行动作 '{action.tool_name}' 时发生异常",
                error_message=str(exc),
            )
            await self._emit_stage(
                "on_action_error",
                {
                    "motive": motive,
                    "chain_step": chain_step,
                    "action": action,
                    "tool_result": error_result,
                    "error": exc,
                },
            )
            return error_result

        after_context = await self._emit_stage(
            "after_action_execute",
            {
                "motive": motive,
                "chain_step": chain_step,
                "action": action,
                "tool_result": result,
            },
        )
        final_result = after_context.get("tool_result", result)
        if final_result.success:
            logger.info(
                f"动作 '{action.tool_name}' 执行完成: success={final_result.success}, summary={final_result.summary}"
            )
        else:
            logger.warning(
                f"动作 '{action.tool_name}' 执行失败: success={final_result.success}, "
                f"summary={final_result.summary}, error={final_result.error_message}"
            )
        return final_result

    async def _enqueue_follow_up_actions(
        self,
        action_queue: List[ActionSpec],
        follow_up_actions: List[ActionSpec],
        motive: str,
        chain_step: int,
    ) -> None:
        if not follow_up_actions:
            return

        context = await self._emit_stage(
            "before_follow_up_enqueue",
            {
                "motive": motive,
                "chain_step": chain_step,
                "action_queue": action_queue,
                "follow_up_actions": list(follow_up_actions),
            },
        )
        action_queue.extend(context.get("follow_up_actions", follow_up_actions))
        await self._emit_stage(
            "after_follow_up_enqueue",
            {
                "motive": motive,
                "chain_step": chain_step,
                "action_queue": action_queue,
            },
        )

    async def _execute_shell_plan(self, motive: str, shell_commands: str):
        """
        将 MainPlanner 产出的 Shell 指令交给 Kernel Interpreter 执行，并记录观察。
        这里只关注“指令 -> 执行结果”链路，真实工具/子规划器由 core 层承接。
        """
        if not shell_commands:
            logger.warning("Planner 没有产出任何指令。")
            return

        results = await self.kernel_interpreter.execute_batch(shell_commands)
        # 将执行结果记入世界模型，供下一轮上一条 observation 使用
        observation = json.dumps(results, ensure_ascii=False)
        self.world_model.set_last_observation(observation)
        self.world_model.add_memory(f"执行指令: {shell_commands.splitlines()} -> {observation}")
        return results

    async def _record_chain_memory(
        self,
        motive: str,
        initial_plan: PlanResult,
        results: List[ToolResult],
        actions: List[ActionSpec],
    ):
        if not results:
            return

        context = await self._emit_stage(
            "before_memory_record",
            {
                "motive": motive,
                "initial_plan": initial_plan,
                "results": results,
                "actions": actions,
            },
        )
        results = context.get("results", results)
        motive = context.get("motive", motive)
        initial_plan = context.get("initial_plan", initial_plan)
        actions = context.get("actions", actions)

        llm_request = self.llm_factory.get_request("planner")
        action_log = ""
        for i, (action, tool_result) in enumerate(zip(actions, results)):
            action_log += (
                f"步骤 {i + 1}: action={action.tool_name}, "
                f"success={tool_result.success}, summary={tool_result.summary}\n"
            )

        prompt = f"""
请根据下面这一轮行动链，提炼一条适合写入短期记忆和长期行为历史的总结。

## 背景
- 当前动机: "{motive}"
- 初始计划原因: "{initial_plan.reason}"
- 动作执行记录:
{action_log}

## 输出要求
1. 总结这轮行动链的结果与影响。
2. 用自然中文表达。
3. 聚焦对后续决策仍有帮助的信息，不要写成流水账。
"""
        final_memory, _ = await llm_request.execute(prompt)

        if final_memory:
            final_memory = final_memory.strip()
            self.world_model.add_memory(final_memory)
            await self._persist_behavior_history(
                motive=motive,
                initial_plan=initial_plan,
                results=results,
                actions=actions,
                final_memory=final_memory,
            )
            await self._emit_stage(
                "after_memory_record",
                {
                    "motive": motive,
                    "initial_plan": initial_plan,
                    "results": results,
                    "actions": actions,
                    "memory": final_memory,
                },
            )

    async def _run_once(self):
        try:

            self.cortex_manager.update_cortices_summaries()
            motive = await self.motive_engine.generate_motive()

            if not motive:
                logger.info("当前没有生成新的动机。")
                return

            self.world_model.motive = motive
            # 刷新任务快照供 Planner 使用
            await self.world_model.refresh_task_snapshots()
            shell_plan = await self.main_planner.plan(motive, self.world_model.get_last_observation())
            await self._execute_shell_plan(motive, shell_plan)
            logger.info(f"动机 '{motive}' 已处理完成。")
        except asyncio.CancelledError:
            logger.warning("AgentLoop 任务被取消。")
            raise
        except Exception as exc:
            logger.error(f"AgentLoop 运行出错: {exc}", exc_info=True)

    async def _run(self):
        self._is_running = True
        logger.info(f"思维循环已启动，心跳间隔 {self.heartbeat_interval} 秒。")
        while self._is_running:
            try:
                await self._run_once()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                logger.info("思维循环已停止。")
                break
            except Exception as exc:
                logger.error(f"思维循环异常: {exc}", exc_info=True)
                await asyncio.sleep(self.heartbeat_interval)

    def start(self):
        if not self._is_running:
            self._main_task = asyncio.create_task(self._run())
        else:
            logger.warning("AgentLoop 已经在运行中。")

    def stop(self):
        if self._is_running and self._main_task:
            self._is_running = False
            self._main_task.cancel()
        else:
            logger.warning("AgentLoop 当前未在运行。")
