import json
import shlex
from typing import Any, Dict, List, Optional

from src.agent.planner.planner_result import PlannerResult
from src.common.di.container import container
from src.common.logger import get_logger
from src.core.task.models import Priority, TaskMode, TaskStatus
from src.core.task.task_manager import TaskManager
from src.core.task.task_store import TaskStore
from src.core.kernel.scheduler import Scheduler
from src.cortex_system import CortexManager

try:
    from src.core.memory import UnifiedMemory
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

logger = get_logger("kernel_interpreter")


class KernelInterpreter:
    """
    内核指令解释器。

    新版能力：
    1. 支持 Planner 的 JSON 输出协议。
    2. 支持 `task create --mode`。
    3. 支持 `task run / task view` 作为主链原子动作协议。
    4. 将旧 `action` 指令降级为兼容占位，不再作为主路径执行模型。
    """

    def __init__(self):
        self.task_manager: TaskManager = container.resolve(TaskManager)
        self.scheduler: Scheduler = container.resolve(Scheduler)
        self.task_store: TaskStore = container.resolve(TaskStore)
        self.cortex_manager: CortexManager = container.resolve(CortexManager)
        self.unified_memory: Optional[UnifiedMemory] = None

        if MEMORY_AVAILABLE:
            try:
                self.unified_memory = container.resolve(UnifiedMemory)
            except Exception:
                pass

    async def execute_batch(self, planner_output: Any) -> List[Dict[str, Any]]:
        """
        批量执行 Planner 输出。

        支持三种输入：
        1. 纯 shell 文本
        2. dict / PlannerResult
        3. JSON 字符串
        """
        parsed_result = self._coerce_planner_output(planner_output)
        results: List[Dict[str, Any]] = []

        for line in parsed_result.shell_commands:
            try:
                cmd_parts = shlex.split(line)
                if not cmd_parts:
                    continue

                res = await self._dispatch(cmd_parts)
                results.append(
                    {
                        "command": line,
                        "status": "success",
                        "output": self._normalize_output(res),
                    }
                )
            except Exception as exc:
                logger.error(f"指令执行失败: {line} | 错误: {exc}")
                results.append({"command": line, "status": "error", "message": str(exc)})

        if parsed_result.thought:
            results.insert(
                0,
                {
                    "command": "__thought__",
                    "status": "info",
                    "output": parsed_result.thought,
                },
            )

        return results

    async def _dispatch(self, args: List[str]) -> Any:
        primary_cmd = args[0].lower()

        if primary_cmd == "task":
            return await self._handle_task_cmd(args[1:])
        if primary_cmd == "memory":
            return await self._handle_memory_cmd(args[1:])
        if primary_cmd == "context":
            return await self._handle_context_cmd(args[1:])
        if primary_cmd == "signal":
            return await self._handle_signal_cmd(args[1:])
        if primary_cmd == "model":
            return await self._handle_model_cmd(args[1:])
        if primary_cmd == "cross_domain":
            return await self._handle_cross_domain_cmd(args[1:])
        if primary_cmd == "skill":
            return await self._handle_skill_cmd(args[1:])
        if primary_cmd == "action":
            return {
                "deprecated": True,
                "message": "action 栈协议已退出主链，请改用 task run / task view / task create --mode。",
            }

        raise ValueError(f"Unknown system call: {primary_cmd}")

    async def _handle_task_cmd(self, args: List[str]) -> Any:
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])

        logger.info(f"任务指令: {sub_cmd} | 参数: {params}")

        if sub_cmd == "create":
            return await self.task_manager.create_task(
                cortex=params.get("cortex", ""),
                target_id=params.get("target", ""),
                priority=self._parse_priority(params.get("pri")),
                motive=params.get("motive", ""),
                mode=self._parse_mode(params.get("mode")),
                context_ref=params.get("context_ref", ""),
                task_config=self._parse_json_or_dict(params.get("config", "{}")),
            )

        if sub_cmd == "exec":
            task_id = params.get("id")
            task = await self.task_store.get(task_id)
            if not task:
                raise ValueError(f"任务不存在: {task_id}")

            dispatch_result = await self.scheduler.dispatch_to_executor(task_id=task_id, entry=None)
            await self.task_manager.update_task_runtime(task_id, increment_execution=True)
            return dispatch_result

        if sub_cmd == "run":
            return await self._handle_task_run(params)

        if sub_cmd == "view":
            return await self._handle_task_view(params)

        if sub_cmd == "suspend":
            return await self.task_manager.suspend_task(params.get("id"))

        if sub_cmd == "mute":
            task_id = params.get("id")
            if task_id:
                return await self.task_store.update_status(task_id, TaskStatus.MUTED)
            return {"success": False, "error": "缺少 id"}

        if sub_cmd == "block":
            return await self.task_manager.block_task(params.get("id"))

        if sub_cmd == "adjust_prio":
            return await self.task_manager.adjust_priority(
                task_id=params.get("id"),
                priority=self._parse_priority(params.get("pri")),
            )

        if sub_cmd == "resume":
            return await self.task_manager.resume_task(params.get("id"))

        if sub_cmd == "kill":
            await self.task_manager.terminate_task(params.get("id"))
            return "terminated"

        raise ValueError(f"未知的任务指令: {sub_cmd}")

    async def _handle_task_run(self, params: Dict[str, str]) -> Any:
        """
        执行一次原子动作。
        """
        cortex = params.get("cortex", "")
        action_name = params.get("action", "")
        if not cortex or not action_name:
            raise ValueError("task run 缺少 cortex 或 action")

        tool_params = self._extract_passthrough_params(
            params,
            reserved={"cortex", "action", "task_id", "target", "pri", "motive", "mode", "config"},
        )
        result = await self.cortex_manager.execute_atomic_action(cortex, action_name, **tool_params)

        task_id = params.get("task_id")
        if task_id:
            await self.task_manager.update_task_runtime(
                task_id,
                last_result={"action": action_name, "result": self._normalize_output(result)},
                append_window={
                    "role": "tool",
                    "content": f"[{cortex}.{action_name}] {self._normalize_output(result)}",
                    "metadata": {"cortex": cortex, "action": action_name},
                },
                increment_execution=True,
            )

        return result

    async def _handle_task_view(self, params: Dict[str, str]) -> Any:
        """
        显式查看控制面板。
        """
        cortex = params.get("cortex", "")
        panel_name = params.get("panel", "")
        if not cortex or not panel_name:
            raise ValueError("task view 缺少 cortex 或 panel")

        panel_params = self._extract_passthrough_params(
            params,
            reserved={"cortex", "panel", "task_id", "target", "pri", "motive", "mode", "config"},
        )
        result = await self.cortex_manager.execute_panel_view(cortex, panel_name, **panel_params)

        task_id = params.get("task_id")
        if task_id:
            task = await self.task_store.get(task_id)
            if task:
                task.cache_view(panel_name, self._normalize_output(result))
                task.append_window_message(
                    role="tool",
                    content=f"[view:{cortex}.{panel_name}] {self._normalize_output(result)}",
                    cortex=cortex,
                    panel=panel_name,
                )
                await self.task_store.save(task)

        return result

    async def _handle_memory_cmd(self, args: List[str]) -> Any:
        if not self.unified_memory:
            return "统一记忆系统未启用"

        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])
        logger.info(f"记忆指令: {sub_cmd} | 参数: {params}")

        if sub_cmd == "store":
            from src.core.memory import MemoryType

            tags_param = params.get("tags", "[]")
            try:
                tags = json.loads(tags_param)
            except Exception:
                tags = tags_param.split(",") if tags_param else []

            memory_id = await self.unified_memory.store(
                content=params.get("content", ""),
                memory_type=MemoryType(params.get("type", "short_term")),
                source_cortex=params.get("cortex", ""),
                source_target=params.get("target", ""),
                tags=tags,
                importance=float(params.get("importance", 0.5)),
                task_id=params.get("task_id", ""),
            )
            return f"已存储记忆: {memory_id}"

        if sub_cmd == "retrieve":
            memories = await self.unified_memory.retrieve(
                query=params.get("query", ""),
                limit=int(params.get("limit", 5)),
                source_cortex=params.get("cortex", ""),
                source_target=params.get("target", ""),
                task_id=params.get("task_id", ""),
                semantic=params.get("semantic", "true").lower() == "true",
            )
            return [
                {
                    "id": m.memory_id,
                    "content": m.content[:50] + "...",
                    "type": m.memory_type.value,
                }
                for m in memories
            ]

        if sub_cmd == "forget":
            memory_id = params.get("id", "")
            success = await self.unified_memory.forget(memory_id)
            return f"{'已' if success else '未能'}遗忘记忆: {memory_id}"

        raise ValueError(f"未知的记忆指令: {sub_cmd}")

    async def _handle_context_cmd(self, args: List[str]) -> Any:
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])

        if sub_cmd == "append":
            task_id = params.get("task_id", "")
            task = await self.task_store.get(task_id)
            if not task:
                return {"success": False, "error": f"任务不存在: {task_id}"}

            task.append_window_message(
                role=params.get("role", "user"),
                content=params.get("content", ""),
                source=params.get("source", "manual"),
            )
            await self.task_store.save(task)
            return {"success": True, "task_id": task_id}

        if sub_cmd == "load":
            task_id = params.get("task_id", "")
            task = await self.task_store.get(task_id)
            if not task:
                return {"success": False, "error": f"任务不存在: {task_id}"}
            return {"task_id": task_id, "task_window": task.task_window, "view_cache": task.view_cache}

        return f"上下文指令 {sub_cmd} 暂未实现"

    async def _handle_signal_cmd(self, args: List[str]) -> Any:
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])
        logger.info(f"信号指令: {sub_cmd} | 参数: {params}")
        return f"信号指令 {sub_cmd} 暂未实现"

    async def _handle_model_cmd(self, args: List[str]) -> Any:
        sub_cmd = args[0].lower()
        return f"模型指令 {sub_cmd} 暂未实现"

    async def _handle_cross_domain_cmd(self, args: List[str]) -> Any:
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])
        logger.info(f"跨域指令: {sub_cmd} | 参数: {params}")
        return f"跨域指令 {sub_cmd} 暂未实现"

    async def _handle_skill_cmd(self, args: List[str]) -> Any:
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])
        logger.info(f"Skill 指令: {sub_cmd} | 参数: {params}")
        return f"Skill 指令 {sub_cmd} 暂未实现"

    def _parse_args(self, arg_list: List[str]) -> Dict[str, str]:
        parsed: Dict[str, str] = {}
        i = 0
        while i < len(arg_list):
            if arg_list[i].startswith("--"):
                key = arg_list[i][2:]
                if i + 1 < len(arg_list) and not arg_list[i + 1].startswith("--"):
                    value = arg_list[i + 1]
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    parsed[key] = value
                    i += 2
                else:
                    parsed[key] = "true"
                    i += 1
            else:
                i += 1
        return parsed

    def _normalize_output(self, res: Any) -> Any:
        if hasattr(res, "to_dict"):
            try:
                return res.to_dict()
            except Exception:
                return str(res)
        return res

    def _parse_priority(self, value: Any) -> Priority:
        if isinstance(value, Priority):
            return value
        if value is None:
            return Priority.LOW

        text = str(value).strip().lower()
        if text in {Priority.CRITICAL.value, "critical"}:
            return Priority.CRITICAL
        if text in {Priority.HIGH.value, "high"}:
            return Priority.HIGH
        if text in {Priority.MEDIUM.value, "medium"}:
            return Priority.MEDIUM
        if text in {Priority.LOW.value, "low"}:
            return Priority.LOW

        try:
            num = float(text)
        except Exception:
            return Priority.LOW

        if num >= 90:
            return Priority.CRITICAL
        if num >= 70:
            return Priority.HIGH
        if num >= 40:
            return Priority.MEDIUM
        return Priority.LOW

    def _parse_mode(self, value: Any) -> TaskMode:
        if isinstance(value, TaskMode):
            return value
        if value is None:
            return TaskMode.ONCE

        text = str(value).strip().lower()
        for mode in TaskMode:
            if mode.value == text:
                return mode
        return TaskMode.ONCE

    def _parse_json_or_dict(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if not value:
            return {}
        try:
            parsed = json.loads(str(value))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _extract_passthrough_params(self, params: Dict[str, str], reserved: set[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in params.items():
            if key in reserved:
                continue
            result[key] = value
        return result

    def _coerce_planner_output(self, planner_output: Any) -> PlannerResult:
        if isinstance(planner_output, PlannerResult):
            return planner_output

        if isinstance(planner_output, dict):
            return PlannerResult(
                thought=str(planner_output.get("thought", "")).strip(),
                shell_commands=self._coerce_shell_commands(planner_output.get("shell_commands", [])),
                raw_content=json.dumps(planner_output, ensure_ascii=False),
                metadata={"format": "dict"},
            )

        if isinstance(planner_output, str):
            text = planner_output.strip()
            if text.startswith("{"):
                try:
                    parsed = json.loads(text)
                    return self._coerce_planner_output(parsed)
                except Exception:
                    pass
            return PlannerResult(
                thought="",
                shell_commands=self._coerce_shell_commands(text),
                raw_content=planner_output,
                metadata={"format": "plain_text"},
            )

        return PlannerResult(raw_content=str(planner_output))

    def _coerce_shell_commands(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [line.strip() for line in value.splitlines() if line.strip()]
        return []
