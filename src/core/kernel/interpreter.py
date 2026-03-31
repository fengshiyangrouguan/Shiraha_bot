import shlex
import asyncio
from typing import List, Dict, Any
from src.common.logger import get_logger
from src.common.di.container import container

# 导入任务管理器和调度器（假设已存在）
# from src.core.task.manager import TaskManager
# from src.core.kernel.scheduler import Scheduler

logger = get_logger("kernel_interpreter")

class KernelInterpreter:
    """
    内核指令解释器 (Kernel Interpreter)
    负责解析并执行来自 MainPlanner 的 Shell 指令集。
    """

    def __init__(self):
        # 通过 DI 获取内核组件
        # self.task_manager: TaskManager = container.resolve(TaskManager)
        # self.scheduler: Scheduler = container.resolve(Scheduler)
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
                
                res = await self.dispatch(cmd_parts)
                results.append({"command": line, "status": "success", "output": res})
            except Exception as e:
                logger.error(f"指令执行失败: {line} | 错误: {e}")
                results.append({"command": line, "status": "error", "message": str(e)})
        
        return results

    async def dispatch(self, args: List[str]) -> Any:
        """
        指令路由分发 (System Call Dispatcher)
        """
        primary_cmd = args[0].lower()

        if primary_cmd == "task":
            return await self._handle_task_cmd(args[1:])
        elif primary_cmd == "idle":
            logger.info("Kernel 进入 IDLE 轮询模式")
            return "system_idle"
        else:
            raise ValueError(f"Unknown system call: {primary_cmd}")

    async def _handle_task_cmd(self, args: List[str]) -> Any:
        """
        处理 task 相关的子指令: create, exec, suspend, kill...
        """
        sub_cmd = args[0].lower()
        params = self._parse_args(args[1:])

        logger.info(f"Kernel Call: task {sub_cmd} | Params: {params}")

        if sub_cmd == "create":
            # 示例: task create --cortex qq --target 12345 --pri 80
            # return await self.task_manager.create_task(
            #     cortex=params.get('cortex'),
            #     target=params.get('target'),
            #     priority=int(params.get('pri', 50))
            # )
            return f"Task created in {params.get('cortex')}"

        elif sub_cmd == "exec":
            # 示例: task exec --id task_01 --entry run_step
            # return await self.scheduler.dispatch_to_executor(
            #     task_id=params.get('id'),
            #     entry=params.get('entry')
            # )
            return f"Executing {params.get('id')}"

        elif sub_cmd == "suspend":
            # 示例: task suspend --id task_01
            # return await self.task_manager.suspend_task(params.get('id'))
            return f"Task {params.get('id')} suspended"

        elif sub_cmd == "kill":
            return f"Task {params.get('id')} terminated"

        else:
            raise ValueError(f"Unknown task sub-command: {sub_cmd}")

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