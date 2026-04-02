import shlex
from typing import List, Dict, Any
from src.common.logger import get_logger
from src.common.di.container import container
from src.core.task.task_manager import TaskManager
from src.core.task.models import Priority, TaskStatus, BaseAction
from src.core.task.task_store import TaskStore
from src.core.kernel.scheduler import Scheduler

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
            # 处理 action 相关指令 (示例: action push --task_id task_01 --action_id act_01 --pri high)
            return await self._handle_action_cmd(args[1:])
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
            if self.task_manager:
                return await self.task_manager.create_task(
                    cortex=params.get('cortex', ''),
                    target_id=params.get('target', ''),
                    priority=self._parse_priority(params.get('pri')),
                    motive=params.get('motive', '')
                )
            return f"对 {params.get('cortex')} 的任务已创建，目标 {params.get('target')}"

        elif sub_cmd == "exec":
            # 示例: task exec --id task_01 --entry run_step
            if self.scheduler:
                return await self.scheduler.dispatch_to_executor(
                    task_id=params.get('id'),
                    entry=params.get('entry')
                )
            return f"注意力转向 {params.get('id')}"

        elif sub_cmd == "suspend":
            # 示例: task suspend --id task_01 
            if self.task_manager:
                return await self.task_manager.suspend_task(params.get('id'))
            return f"任务 {params.get('id')} 已挂起"
        
        elif sub_cmd == "mute":
            # 示例: task mute --id task_01
            task_id = params.get('id')
            if self.task_store and task_id:
                return await self.task_store.update_status(task_id, TaskStatus.MUTED)
            return f"任务 {params.get('id')} 已静默"

        elif sub_cmd == "block":
            # 示例: task block --id task_01
            if self.task_manager:
                return await self.task_manager.block_task(params.get('id'))
            return f"任务 {params.get('id')} 已阻塞"

        elif sub_cmd == "adjust_prio":
            # 示例: task adjust_prio --id task_01 --pri high
            if self.task_manager:
                return await self.task_manager.adjust_priority(
                    task_id=params.get('id'),
                    priority=self._parse_priority(params.get('pri'))
                )
            return f"任务 {params.get('id')} 已调整优先级"
        
        elif sub_cmd == "resume":
            # 示例: task resume --id task_01
            if self.task_manager:
                return await self.task_manager.resume_task(params.get('id'))
            return f"任务 {params.get('id')} 已恢复"
        
        elif sub_cmd == "kill":
            if self.task_manager:
                await self.task_manager.terminate_task(params.get('id'))
                return "terminated"
            return f"任务 {params.get('id')} 已结束"

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
            # 示例: action push --task_id task_01 --action_id act_01 --action_name small_talk --pri high
            task_id = params.get('task_id')
            action_id = params.get('action_id')
            action_name = params.get('action_name')
            priority = self._parse_priority(params.get('pri'))
            if self.task_store and task_id and action_id:
                task = await self.task_store.get(task_id)
                if task:
                    task.add_action(BaseAction(action_id=action_id, priority=priority))
                    await self.task_store.save(task)
                    return f"行为 {action_id} 已添加到任务 {task_id}"
            return f"无法添加行为 {action_id} 到任务 {task_id}"

    def _parse_args(self, arg_list: List[str]) -> Dict[str, str]:
        """
        简单的命令行参数解析器 (--key value)
        """
        parsed = {}
        i = 0
        while i < len(arg_list):
            if arg_list[i].startswith("--"):
                key = arg_list[i][2:]
                if i + 1 < len(arg_list):
                    parsed[key] = arg_list[i+1]
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
