import json
from typing import Optional

from src.common.di.container import container
from src.agent.world_model import WorldModel
from src.agent.planner.planner_result import PlanResult
from src.cortices.manager import CortexManager

from src.llm_api.dto import LLMMessageBuilder
from src.llm_api.factory import LLMRequestFactory
from src.llm_api.plan_parser import PlanParser

class MainPlanner:
    def __init__(self):
        self.world_model: WorldModel = container.resolve(WorldModel)
        self.cortex_manager: CortexManager = container.resolve(CortexManager)
        llm_factory: LLMRequestFactory = container.resolve(LLMRequestFactory)

        self.llm_request = llm_factory.get_request("main_planner")
        self.plan_parser = PlanParser(logger_name="main_planner")

    async def plan(self, motive: str, previous_observation: str = None) -> str:
        context = self.world_model.get_full_system_state() 
        
        # 这里的 system_prompt 彻底 CLI 化
        system_prompt = f"""
## 身份定位：Agent OS Kernel Shell
你是一个高性能 AI 操作系统内核。你的唯一任务是输出 **Shell 指令** 来调度系统资源。

## 核心指令集 (Man Pages):
- `task create --cortex <name> --target <id> --pri <0-100>` : 初始化任务。
- `task suspend --id <id>` : 挂起指定任务，保存上下文。
- `task resume --id <id>` : 恢复被挂起的任务。
- `task exec --id <id> --entry <method>` : 启动任务的子规划器执行具体逻辑。
- `task focus --id <id>` : 将系统注意力锁移至目标任务。
- `task kill --id <id>` : 彻底销毁任务。
- `idle` : 当没有任何高优任务或等待时，进入空闲轮询。

## 运行约束:
1. **禁止输出 JSON**，禁止输出 Markdown，禁止任何自然语言解释。
2. **仅输出一行或多行指令**。
3. 优先级逻辑：{context['notifications']} 中的紧急信号必须优先处理。
4. 所有的数据交互通过 `--target` ID 引用，不要在指令中包含具体聊天文案。

## 示例输出:
task suspend --id task_01
task create --cortex qq --target msg_99 --pri 90
task focus --id task_99
task exec --id task_99 --entry auto_reply
"""

        user_prompt = f"""
[SYSTEM_STATE]
MOTIVE: {motive}
ACTIVE_TASKS: {json.dumps(context['active_tasks'])}
INTERRUPTS: {context['notifications']}
LAST_OBS: {previous_observation or "BOOT_SUCCESS"}

[INPUT_SHELL]
# 请输出下一步执行的指令：
"""

        # 调用 LLM，注意这里不需要 json.dumps(prompt_dict)，直接传消息列表
        builder = LLMMessageBuilder()
        builder.add_system_message(system_prompt)
        builder.add_user_message(user_prompt)
        
        # 假设你的 execute 现在直接返回 content 字符串
        content, _ = await self.llm_request.execute(messages=builder.get_message_dict())
        
        # 直接返回 shell 字符串，交给外部的 Kernel Interpreter 解析
        return content.strip()