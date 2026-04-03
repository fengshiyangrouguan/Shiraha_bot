import shlex
import json
from typing import List, Dict, Any
from src.common.logger import get_logger
from src.common.di.container import container
from src.core.task.task_manager import TaskManager
from src.core.task.models import Priority, TaskStatus, BaseAction
from src.core.task.task_store import TaskStore
from src.core.kernel.scheduler import Scheduler

try:
    from src.core.memory import UnifiedMemory
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

logger = get_logger("kernel_interpreter")


class KernelInterpreter:
    """
    内核指令解释器 (Kernel Interpreter)
    负责解析并执行来自 MainPlanner 的 Shell 指令集。
    """

    def __init__(self):
        self.task_manager: TaskManager = container.resolve(TaskManager)
        self.scheduler: Scheduler = container.resolve(Scheduler)
        self.task_store: TaskStore = container.resolve(TaskStore)
        self.unified_memory: UnifiedMemory = None

        # 尝试获取统一记忆系统
        if MEMORY_AVAILABLE:
            try:
                self.unified_memory = container.resolve(UnifiedMemory)
            except Exception:
                pass

    async def execute_batch(self, shell_commands: str) -> List[Dict[str, Any]]:
        """
        批量执行多行指令，支持 && 或 换行符
        """
        results = []
        # 处理多行指令
        lines = [line.strip() for line in shell_commands.split('\n') if line.strip()]

        for line in lines:
            try:
                # 使用 shlex 像 bash 一样解析字符串，自动处理引号和空格
                cmd_parts = shlex.split(line)
                if not cmd_parts:
                    continue

                res = await self._dispatch(cmd_parts)
                results.append({"command": line, "status": "success", "output": self._normalize_output(res)})
            except Exception as e:
                logger.error(f"指令执行失败: {line} | 错误: {e}")
                results.append({"command": line, "status": "error", "message": str(e)})

        return results

    async def _dispatch(self, args: List[str]) -> Any:
        """
        指令路由分发 (System Call Dispatcher)
        """
        primary_cmd = args[0].lower()

        if primary_cmd == "task":
            return await self._handle_task_cmd(args[1:])
        elif primary_cmd == "action":
            return await self._handle_action_cmd(args[1:])
        elif primary_cmd == "memory":
            return await self._handle_memory_cmd(args[1:])
        elif primary_cmd == "context":
            return await self._handle_context_cmd(args[1:])
        elif primary_cmd == "signal":
            return await self._handle_signal_cmd(args[1:])
        elif primary_cmd == "model":
            return await self._handle_model_cmd(args[1:])
        elif primary_cmd == "cross_domain":
            return await self._handle_cross_domain_cmd(args[1:])
        elif primary_cmd == "skill":
            return await self._handle_skill_cmd(args[1:])
        else:
            raise ValueError(f"Unknown system call: {primary_cmd}")

    async def _handle_task_cmd(self, args: List[str]) -> Any:
        """
        处理 task 相关的子指令: create, exec, suspend, kill...
        """
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])

        logger.info(f"任务指令: {sub_cmd} | 参数: {params}")

        if sub_cmd == "create":
            # 示例: task create --cortex qq --target 12345 --pri low --motive "闲聊"
            return await self.task_manager.create_task(
                cortex=params.get('cortex', ''),
                target_id=params.get('target', ''),
                priority=self._parse_priority(params.get('pri')),
                motive=params.get('motive', '')
            )

        elif sub_cmd == "exec":
            return await self.scheduler.dispatch_to_executor(
                task_id=params.get('id'),
                entry=params.get('entry')
            )

        elif sub_cmd == "suspend":
            return await self.task_manager.suspend_task(params.get('id'))

        elif sub_cmd == "mute":
            task_id = params.get('id')
            if task_id:
                return await self.task_store.update_status(task_id, TaskStatus.MUTED)
            return f"任务 {params.get('id')} 已静默"

        elif sub_cmd == "block":
            return await self.task_manager.block_task(params.get('id'))

        elif sub_cmd == "adjust_prio":
            return await self.task_manager.adjust_priority(
                task_id=params.get('id'),
                priority=self._parse_priority(params.get('pri'))
            )

        elif sub_cmd == "resume":
            return await self.task_manager.resume_task(params.get('id'))

        elif sub_cmd == "kill":
            await self.task_manager.terminate_task(params.get('id'))
            return "terminated"

        else:
            raise ValueError(f"未知的任务指令: {sub_cmd}")

    async def _handle_action_cmd(self, args: List[str]) -> Any:
        """
        处理 action 相关的子指令: push, pop, complete...
        """
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])

        logger.info(f"行为指令: {sub_cmd} | 参数: {params}")

        if sub_cmd == "push":
            # 示例: action push --task_id task_01 --action_id act_01 --action_type sequential --skill chat --steps [...] --pri high
            from src.core.action import create_action

            task_id = params.get('task_id')
            action_id = params.get('action_id')
            action_type = params.get('action_type', 'generic')
            skill_name = params.get('skill', '')
            priority = self._parse_priority(params.get('pri'))

            if not task_id:
                return "错误: 缺少 task_id"

            # 解析额外参数
            extra_params = {}
            if action_type == 'sequential':
                steps = params.get('steps', '[]')
                try:
                    extra_params['steps'] = json.loads(steps)
                except json.JSONDecodeError:
                    extra_params['steps'] = steps.split(',')
            elif action_type == 'blocking':
                extra_params['wait_criteria'] = params.get('wait_criteria', '')
                extra_params['timeout'] = float(params.get('timeout', 300))
            elif action_type == 'single_step':
                extra_params['command'] = params.get('command', '')
                cmd_params = params.get('params', '{}')
                try:
                    extra_params['command_params'] = json.loads(cmd_params)
                except json.JSONDecodeError:
                    pass

            # 创建 Action
            action = create_action(
                action_type=action_type,
                skill_name=skill_name,
                action_id=action_id,
                priority=priority,
                **extra_params
            )

            # 添加到任务
            if action_id:
                extra_params['action_id'] = action_id

            task = await self.task_store.get(task_id)
            if task:
                task.add_action(action)
                await self.task_store.save(task)
                return f"行为 {action.action_id} 已添加到任务 {task_id}"
            return f"无法添加行为到任务 {task_id}"

        elif sub_cmd == "complete":
            # 示例: action complete --task_id task_01 --result "完成"
            task_id = params.get('task_id')
            result = params.get('result', '')
            task = await self.task_store.get(task_id)
            if task and task.actions:
                task.actions[-1].finalize(result)
                return f"任务 {task_id} 的顶层行为已完成"
            return f"无法完成任务 {task_id}"

        else:
            raise ValueError(f"未知的行为指令: {sub_cmd}")

    async def _handle_memory_cmd(self, args: List[str]) -> Any:
        """
        处理 memory 相关的子指令: store, retrieve, forget
        """
        if not self.unified_memory:
            return "统一记忆系统未启用"

        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])

        logger.info(f"记忆指令: {sub_cmd} | 参数: {params}")

        if sub_cmd == "store":
            # 示例: memory store --content "..." --tags[qq,complaint] --importance 0.8 --cortex qq --target user123
            from src.core.memory import MemoryType

            tags_param = params.get('tags', '[]')
            try:
                tags = json.loads(tags_param)
            except:
                tags = tags_param.split(',') if tags_param else []

            memory_id = await self.unified_memory.store(
                content=params.get('content', ''),
                memory_type=MemoryType(params.get('type', 'short_term')),
                source_cortex=params.get('cortex', ''),
                source_target=params.get('target', ''),
                tags=tags,
                importance=float(params.get('importance', 0.5)),
                task_id=params.get('task_id', '')
            )
            return f"已存储记忆: {memory_id}"

        elif sub_cmd == "retrieve":
            # 示例: memory retrieve --query "..." --limit 5
            memories = await self.unified_memory.retrieve(
                query=params.get('query', ''),
                limit=int(params.get('limit', 5)),
                source_cortex=params.get('cortex', ''),
                source_target=params.get('target', ''),
                task_id=params.get('task_id', ''),
                semantic=params.get('semantic', 'true').lower() == 'true'
            )
            return [{"id": m.memory_id, "content": m.content[:50] + "...", "type": m.memory_type.value}
                    for m in memories]

        elif sub_cmd == "forget":
            # 示例: memory forget --id mem_xxx
            memory_id = params.get('id', '')
            success = await self.unified_memory.forget(memory_id)
            return f"{'已' if success else '未能'}遗忘记忆: {memory_id}"

        else:
            raise ValueError(f"未知的记忆指令: {sub_cmd}")

    async def _handle_context_cmd(self, args: List[str]) -> Any:
        """
        处理 context 相关的子指令: load, append
        """
        # TODO: 实现上下文加载和追加
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])
        return f"上下文指令 {sub_cmd} 暂未实现"

    async def _handle_signal_cmd(self, args: List[str]) -> Any:
        """
        处理 signal 相关的子指令: emit, broadcast
        """
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])

        logger.info(f"信号指令: {sub_cmd} | 参数: {params}")

        # TODO: 实现 Signal 系统
        if sub_cmd == "emit":
            # 示例: signal emit --target user123 --cortex qq --type complaint --content "..."
            return f"向 {params.get('target', '')} 发出 {params.get('type', '')} 信号"

        elif sub_cmd == "broadcast":
            # 示例: signal broadcast --type "need_help" --content "..."
            return f"广播 {params.get('type', '')} 信号"

        else:
            raise ValueError(f"未知的信号指令: {sub_cmd}")

    async def _handle_model_cmd(self, args: List[str]) -> Any:
        """
        处理 model 相关的子指令: select
        """
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])

        # TODO: 实现动态模型选择
        return f"模型指令 {sub_cmd} 暂未实现"

    async def _handle_cross_domain_cmd(self, args: List[str]) -> Any:
        """
        处理跨域相关指令: request, transfer
        """
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])

        logger.info(f"跨域指令: {sub_cmd} | 参数: {params}")

        # TODO: 实现跨域机制
        if sub_cmd == "request":
            # 示例: cross_domain request --from task_a --to task_b --payload "..."
            return f"从 {params.get('from', '')} 向 {params.get('to', '')} 发起跨域请求"

        elif sub_cmd == "transfer":
            # 示例: cross_domain transfer --content "..." --from source --to target
            return f"从 {params.get('from', '')} 向 {params.get('to', '')} 转移内容"

        else:
            raise ValueError(f"未知的跨域指令: {sub_cmd}")

    async def _handle_skill_cmd(self, args: List[str]) -> Any:
        """
        处理 skill 相关指令: modify, analyze, update, evolve
        """
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])

        logger.info(f"Skill 指令: {sub_cmd} | 参数: {params}")

        # TODO: 实现 Skill 自我修改系统
        if sub_cmd == "modify":
            # 示例: skill modify --cortex qq --file chat.md --content "..."
            return f"修改 {params.get('cortex', '')} 的 skill: {params.get('file', '')}"

        elif sub_cmd == "analyze":
            # 示例: skill analyze --cortex qq --file chat.md --action_name complaint
            return f"分析 {params.get('cortex', '')}/{params.get('file', '')} 的 {params.get('action_name', '')} 行为"

        elif sub_cmd == "update":
            # 示例: skill update --cortex qq --file chat.md --pattern "pattern" --description "..."
            return f"更新 {params.get('cortex', '')} 的 skill"

        elif sub_cmd == "evolve":
            # 示例: skill evolve --based_on_interactions --timeframe "7d"
            return f"基于互动演化 skill（时间范围: {params.get('timeframe', '7d')}）"

        else:
            raise ValueError(f"未知的 skill 指令: {sub_cmd}")

    def _parse_args(self, arg_list: List[str]) -> Dict[str, str]:
        """
        简单的命令行参数解析器 (--key value)
        支持值包含空格（使用引号）
        """
        parsed = {}
        i = 0
        while i < len(arg_list):
            if arg_list[i].startswith("--"):
                key = arg_list[i][2:]
                if i + 1 < len(arg_list):
                    value = arg_list[i+1]
                    # 移除引号（移除外层的单引号或双引号）
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    parsed[key] = value
                    i += 2
                else:
                    parsed[key] = True
                    i += 1
            else:
                i += 1
        return parsed

    def _normalize_output(self, res: Any) -> Any:
        """将执行结果转换为可序列化形态。"""
        if hasattr(res, "to_dict"):
            try:
                return res.to_dict()
            except Exception:
                return str(res)
        return res

    def _parse_priority(self, value: Any) -> Priority:
        """
        将外部参数解析为 Priority 枚举。
        兼容两种输入：
        1) 枚举名/值字符串：critical/high/medium/low
        2) 传统数字刻度：0~100
        """
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
