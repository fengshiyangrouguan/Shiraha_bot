# src/agent/planners/main_planner.py
import json
from typing import List, Dict, Any

# 导入我们新的 BasePlanner 和 DTOs
from .base_planner import BasePlanner
from src.llm_api.dto import ToolCall, LLMMessageBuilder
from src.agent.world_model import WorldModel
from src.features.manager import FeatureManager
from src.system.di.container import container

# 确保通用工具在启动时被加载和注册
import src.features.common_tools

class MainPlanner(BasePlanner):
    """
    主规划器。
    继承自 BasePlanner，负责将高阶意图分解为具体的行动步骤。
    """
    def __init__(self):
        # 调用父类的构造函数，并传入自己的任务名称
        super().__init__(task_name="main_planning")
        self.feature_manager:FeatureManager = container.resolve(FeatureManager)

    def _build_prompt(self, motive: str, world_model: WorldModel) -> str:
        """
        [实现] 构建 ReAct (Reason+Act) 风格的提示。
        这个方法覆盖了 BasePlanner 中的抽象方法。
        """
        context = world_model.get_context_for_motive()
        available_tools =self.feature_manager.get_tool_schemas(scope="main")


        system_prompt = (
            "你是一个高度智能的AI代理的“主规划器”。你的任务是接收一个高阶意图，并决定执行哪个工具来以最有效的方式推进这个意图。"
            "请严格遵循 ReAct (Reason+Act) 的思考模式：\n"
            "1. **思考 (Reasoning)**: 分析当前意图、世界状态和可用工具，阐述你的思考过程和决策依据。\n"
            "2. **行动 (Action)**: 从可用工具列表中选择一个最合适的工具，并给出调用它所需的具体参数。\n"
            "你的输出必须是一个严格的 JSON 对象，格式如下：\n"
            "{\"thought\": \"你的思考过程...\", \"action\": {\"tool_name\": \"工具名称\", \"parameters\": {\"参数名\": \"参数值\"}}}"
        )
        
        user_prompt = (
            f"## 当前高阶意图:"
            f"\"{motive}\"\n\n"
            f"## 你的核心身份与当前状态:\n"
            f"- 名字: {context['bot_name']}\n"
            f"- 性格: {context['bot_personality']}\n"
            f"- 当前情绪: {context['mood']}\n\n"
            f"## 世界状态与近期活动:\n"
            f"- 当前时间: {context['time']}\n"
            f"{context['alert']}\n"
            f"\n- 近期活动总结:\n{context['action_summary']}\n\n"
            f"## 可用工具列表:\n"
            f"{json.dumps(obj=available_tools, ensure_ascii=False, indent=2)}\n\n"
            "---"
            "基于以上信息，进行你的下一步“思考”和“行动”。"
        )
        
        return f"{system_prompt}\n\n{user_prompt}"