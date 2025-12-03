# src/cortices/qq_chat/deep_chat_planner.py
import json
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime

from src.agent.planner.base_planner import BasePlanner
from src.agent.world_model import WorldModel
from src.common.logger import get_logger
from ..chat_stream import QQChatStream
from ..qq_chat_data import QQChatData
from src.llm_api.dto import LLMMessageBuilder

if TYPE_CHECKING:
   from src.cortices.manager import CortexManager

logger = get_logger("QQDeepChatSubPlanner")

class DeepChatSubPlanner(BasePlanner):
    """
    负责执行深度聊天任务的 SubPlanner。
    继承自 BasePlanner，使用 LLM 进行 ReAct 风格的规划。
    """
    def __init__(self,
                 cortex_manager: "CortexManager",
                 conversation_id: str,
                 tool_scope: str):
        """
        初始化 DeepChatSubPlanner。

        Args:
            cortex: 调用此 SubPlanner 的 QQChatCortex 实例。
            conversation_id: 当前任务关联的对话ID，用于从 WorldModel 中定位上下文。
            tool_scope: 此 SubPlanner 运行时可以使用的工具作用域。
        """
        super().__init__("main_planner","QQDeepChatSubPlanner") # 假设 config.toml 中定义了 deep_chat_planner_task
        self.cortex_manager = cortex_manager
        self.conversation_id = conversation_id
        self.tool_scope = tool_scope
        self.last_obs_time_mark = 0.0 # 模仿 MaiBot 的时间戳标记，用于上下文切片


    def build_prompt(self) -> List[Dict[str, Any]]:
        """
        构建特定于深度聊天规划器的 LLM 提示。
        """
        messages: List[Dict[str, Any]] = []

        # 1. 获取聊天历史和相关上下文
        qq_chat_data:QQChatData = self.world_model.get_cortex_data("qq_chat")
        chat_stream:QQChatStream = qq_chat_data.get_stream_by_id(self.conversation_id)
        chat_history = chat_stream.build_chat_history_for_llm()
        
        # 2. 构建 Prompt
        context:Dict[str, Any] = self.world_model.get_context_for_motive()
        available_tool_list = self.cortex_manager.get_tool_schemas(self.tool_scope)
        if chat_stream.conversation_info.conversation_type == "group":
          chat_target = f"你正在群聊{chat_stream.conversation_info.conversation_name}中与群友聊天。"
        else:
          chat_target = f"你正在与用户{chat_stream.conversation_info.conversation_name}进行私聊。"
        system_prompt = (
            f"## 你的身份设定与当前状态:\n"
            f"- **你的名字**： {context['bot_name']}\n"
            f"- **你的性格**： {context['bot_personality']}\n"
            f"- **你的兴趣**： {context['bot_interest']}\n"
            f"- **你的当前情绪**： \n{context['mood']}\n\n"  
            f"## 核心规则：ReAct 思考模式\n"

            "请严格遵循 ReAct (Reason+Act) 的思考模式：\n"
            "1. **思考 (Reasoning)**: 分析当前意图、世界状态和可用工具，阐述你的思考过程和决策依据。\n"
            "2. **行动 (Action)**: 从可用行动列表中按行动顺序选择一个或多个合适的工具，并给出调用它所需的具体参数。\n"
            "你的输出必须是一个**严格的 JSON 对象**，且必须包含 'thought' 字段和 'actions' 字段。\n"
            "理由(thought)要求是一段精简的平文本，不要分点。"
        )
        user_prompt = (
            f"## 1. 世界状态与近期活动\n"
            f"现在是 {context['time']}。\n"
            f"- **重要通知**: {context['alert']}\n"  
            f"## 2. 聊天内容\n"
            f"{chat_target} 以下是聊天内容：\n"
            f"{chat_history}\n\n"
            f"## 3. 可用的工具列表: \n"
            "```json\n"
            f"{json.dumps(available_tools, ensure_ascii=False, indent=2)}"
            "```\n\n"
            f"## 决策指令\n"
            "你正在与别人进行深入的对话，请根据上述上下文信息、你的身份和当前状态、可用的工具，决定下一步的行动。请严格遵循 ReAct 模式。"
            f"记住，你的回答必须是一个 JSON 对象，格式如下所示。"
            f"--- 严格输出 JSON 格式 ---\n"
            f"```json\n"
            f"{{\n"
            f"    \"thought\": \"你的思考过程...\",\n"
            f"    \"actions\": [\n"
            f"        {{\n"
            f"            \"tool_name\": \"工具名\",\n"
            f"            \"parameters\": {{...}}\n"
            f"            \"reason\": \"触发原因\",\n"
            f"        }}\n"
            f"        // ... 可选的多个Action\n"
            f"    ]\n"
            f"}}\n"
            f"```\n"


        )
        builder = LLMMessageBuilder()

        builder.add_system_message(system_prompt)
        builder.add_user_message(user_prompt)
        
        self.prompt = builder.get_message_dict()
