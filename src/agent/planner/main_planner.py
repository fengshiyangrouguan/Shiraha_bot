# src/agent/planner/main_planner.py
import json
from typing import List, Dict, Any

from .base_planner import BasePlanner
from src.llm_api.dto import LLMMessageBuilder
from src.agent.world_model import WorldModel
from src.cortices.manager import CortexManager
from src.system.di.container import container

class MainPlanner(BasePlanner):
    """
    主规划器。
    负责将高阶意图分解为具体的行动步骤。
    它声明自己需要一个 `motive` 字符串作为输入。
    """
    def __init__(self):
        super().__init__(task_name="main_planner", logger_name="MainPlanner")
        self.cortex_manager: CortexManager = container.resolve(CortexManager)

    async def plan(self, motive:str, previous_observation:str = None) -> List[Dict[str, Any]]:
        """
        [实现] 构建 ReAct (Reason+Act) 风格的提示,然后send_to_LLM
        """
        context = self.world_model.get_context_for_motive()
        available_tools = self.cortex_manager.get_tool_schemas(scope="main")

        system_prompt = (
            f"## 你的身份设定与当前状态:\n"
            f"**你的名字**: {context['bot_name']}\n"
            f"**你的性格**: {context['bot_personality']}\n"
            f"**你的兴趣**: {context['bot_interest']}\n\n"
            f"**你的当前情绪**: \n{context['mood']}\n\n"  
            f"**你的当前动机意图**: \"{motive}\"\n\n"
            f"## 核心规则：ReAct 思考模式\n"
            "请严格遵循 ReAct (Reason+Act) 的思考模式：\n"
            "1. **思考 (Reasoning)**: 分析当前意图、世界状态和可用工具，阐述你的思考过程和决策依据。\n"
            "2. **行动 (Action)**: 从可用工具列表中选择一个最合适的工具，并给出调用它所需的具体参数。\n"
            "你的输出必须是一个严格的 JSON 对象，格式如下：\n"
            f"```json\n"
            "{\"thought\": \"你的思考过程...\", \"action\": {\"tool_name\": \"工具名称\", \"parameters\": {\"参数名\": \"参数值\"}}}"
            f"```\n"
            "如果：1.没有合适的工具，2.意图已完成，3.你认为当前意图不需要任何操作时，请在action中返回 `{\"tool_name\": \"finish\", \"parameters\": {}}`。"
            "理由(thought)要求是一段精简的平文本，不要分点。"
        )
        
        user_prompt = (
            f"## 1. 世界状态与近期活动\n"
            f"**当前时间**: {context['time']}\n"          
            f"**重要通知**: {context['alert']}\n"  
            f"**近期活动总结**:\n{context['action_summary']}\n\n"
            f"{previous_observation}\n\n"
            f"## 2. 可用的工具列表: \n"
            "```json\n"
            f"{json.dumps(available_tools, ensure_ascii=False, indent=2)}"
            "```\n\n"
            f"--- \n"
            "**基于以上信息，开始思考并输出你下一步的 JSON 决策（包含`thought`和`action`）。**"
        )
        
        builder = LLMMessageBuilder()
        builder.add_system_message(system_prompt)
        builder.add_user_message(user_prompt)
        
        prompt = builder.get_message_dict()
        plan_result = await self.send_to_LLM(prompt)
        return plan_result
        
    

#            "你是一个高度智能的AI代理的“主规划器”。你的任务是接收一个高阶意图，并决定执行哪个工具来以最有效的方式推进这个意图。"