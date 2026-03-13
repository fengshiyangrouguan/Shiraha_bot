import json
from typing import Dict, Any, Optional, TYPE_CHECKING, List

from src.common.action_model.action_spec import ActionSpec
from src.common.action_model.tool_result import ToolResult
from src.cortices.tools_base import BaseTool
from src.agent.world_model import WorldModel
from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter
from src.cortices.qq_chat.data_model.qq_chat_data import QQChatData
from src.cortices.qq_chat.data_model.chat_stream import QQChatStream
from src.common.event_model.info_data import ConversationInfo
from src.common.database.database_manager import DatabaseManager
from src.llm_api.factory import LLMRequestFactory
from src.common.di.container import container
from src.common.logger import get_logger
from src.extensions.social_damper import SocialDamper
from src.cortices.qq_chat.cortex import QQChatCortex
from src.cortices.qq_chat.chat.replyer import QQReplyer
from src.cortices.qq_chat.chat.deep_chat_sub_agent import DeepChatSubAgent

if TYPE_CHECKING:
    
    from src.cortices.manager import CortexManager

logger = get_logger("qq_chat")

class BatchQuickPlanTool(BaseTool):
    """
    批量规划多个会话的行动意图，并生成一个后续需要执行的动作列表。
    """
    def __init__(self, adapter: QQNapcatAdapter, cortex:"QQChatCortex",cortex_manager: "CortexManager"):
        super().__init__(cortex_manager)
        self.adapter = adapter
        self.cortex = cortex
        self._world_model = container.resolve(WorldModel)
        self.llm_request_factory = container.resolve(LLMRequestFactory)
        self.database_manager = container.resolve(DatabaseManager)
        self.social_damper: SocialDamper = container.resolve(SocialDamper)

        self.replyer = QQReplyer(world_model=self._world_model,adapter=self.adapter,llm_request_factory=self.llm_request_factory,database_manager=self.database_manager,cortex=self.cortex)
        self.deep_chat_agent = DeepChatSubAgent(world_model=self._world_model,adapter=self.adapter,llm_request_factory=self.llm_request_factory,database_manager=self.database_manager,cortex=self.cortex,replyer=self.replyer)
        
        
    @property
    def scope(self) -> str:
        return "qq_app"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "batch_quick_plan",
            "description": "对多个会话的当前状态进行快速分析，并规划出下一步的行动意图列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "replies": {
                        "type": "array",
                        "description": "需要进行规划的会话列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "conversation_id": {"type": "string", "description": "目标聊天 ID"},
                                "intent": {"type": "string", "description": "对这个会话的初步行动意图"},
                            },
                        }
                    },
                }
            },
            "required": ["replies"]
        }

    async def _build_prompt(self, conversation_info: ConversationInfo, intent: str, history: str):
        """构建提示词"""
        name = self._world_model.bot_name
        personality = self._world_model.bot_personality
        interest = self._world_model.bot_interest
        mood = self._world_model.mood
        # plan_style = self._world_model.bot_plan_style
        time = self._world_model.get_current_time_string()
        damper_result:dict = await self.social_damper.damp_intent(intent,history)
        if damper_result["should_damp"]:
            intent_final=f"原始动机在当前语境突兀，社交阻尼器已介入。新动机规划为“{damper_result["damped_intent"]}”"
            logger.info(f"社交阻尼器介入修正动机：{damper_result["damped_intent"]}")
        else:
            intent_final=f"你当前的行动意图是：{intent}"



        if conversation_info.conversation_type == "group":
            chat_target = f"你正在群聊中与群友聊天。"
        else:
            chat_target = f"你正在与用户{conversation_info.conversation_name}进行私聊。"

# - **你的兴趣**： {interest}
        prompt = (
f"""
## 你的身份设定与当前状态:
- **你的名字**： {name}
- **你的性格**： {personality}

- **你的当前情绪**： {mood}

{time}
{chat_target}

## **以下是最近聊天记录**,请仔细阅读：
{history}

## 待执行的社交规划
{intent_final}

## 可用的 Action 规范
- **去重检查**：不要重复去执行你近期活动中已干过的事。
- **关联性**：若聊天内容与你无关，优先选择 no_reply。

1. **reply**: 进行快速回复然后关掉qq去干别的事情
{{
    "action": "reply",
    "parameters":{
        "reason": "回复的原因/意图"
    }
}}

2. **no_reply**: 保持沉默，不参与该会话。
{{
    "action": "no_reply",
    "parameters":{
        "reason": "沉默的理由"
    }

}}



输出格式示例如下：
{{
    "action": "行动名称",
    "parameters":{{<需要的参数>}}
}}

- **只输出 JSON 代码块**，不要任何多余文字。
请基于这些内容，选择一个action，生成JSON输出。
""")
        return prompt

    async def execute(self, replies: list[Dict[str, Any]], **kwargs) -> ToolResult:
        """
        批量规划，并返回一个包含 ActionSpec 列表的 ToolResult。
        """
        planned_actions: List[ActionSpec] = []
        summary_logs: List[str] = []

        try:
            qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
            
            for item in replies:
                conversation_id = item.get("conversation_id")
                intent = item.get("intent")
                
                # 1. 获取会话上下文
                chat_stream: QQChatStream = await qq_chat_data.get_or_create_stream_by_id(conversation_id)
                if not chat_stream:
                    summary_logs.append(f"找不到 {conversation_id} 的会话记录，已跳过。")
                    continue
                
                conversation_info = chat_stream.conversation_info
                recent_messages = chat_stream.build_chat_history_has_msg_id()
                logger.info(f"正在为'{conversation_info.conversation_name}'规划意图：{intent}")
                prompt = self._build_prompt(conversation_info,intent,recent_messages)
                
                llm_factory = self.llm_request_factory
                llm_request = llm_factory.get_request("planner")
                content, _ = await llm_request.execute(prompt=prompt)

                llm_factory = self.llm_request_factory
                llm_request = llm_factory.get_request("planner")
                content, _ = await llm_request.execute(prompt=prompt)
                logger.info(f"原始规划：{content}")

                # 4. 解析生成的JSON
                try:
                    json_str = content.strip()
                    if json_str.startswith("```"):
                        json_str = json_str.replace("```json", "").replace("```", "").strip()
                    action = json.loads(json_str)

                    #校验一下是否为有效的字典
                    if not isinstance(action, dict):
                        raise ValueError("输出格式非 JSON 对象")
                    
                    act_name = action.get("action")
                    parameters = action.get("parameters")
                    action = ActionSpec(act_name,parameters)
                    planned_actions.append(action)

                    reason = parameters.get("reason", "无")
                    logger.info(f"执行动作: {act_name}, 理由: {reason}")

                    if act_name == "no_reply":
                        logger.info(f"我决定不回复: {reason}")
                        summary_logs.append(f"在“{conversation_info.conversation_name}”聊天中，我决定不参与聊天，原因是{reason}")
                    elif act_name == "reply":
                        result = await self.replyer.execute(reason,chat_stream)
                        summary_logs.append(f"在“{conversation_info.conversation_name}”聊天中，{result}")
                        # 执行回复发送...
                    elif act_name == "deep_chat":
                        deep_chat_tool_result:ToolResult = await self.deep_chat_agent.run(reason,chat_stream)
                        summary = deep_chat_tool_result.summary
                        summary_logs.append(f"在“{conversation_info.conversation_name}”聊天中，{summary}")
                        if deep_chat_tool_result.follow_up_action:
                            planned_actions.append(deep_chat_tool_result.follow_up_action)

                except Exception as e:
                    summary_logs.append(f"在“{conversation_info.conversation_name}”聊天中，行动解析失败了，报错:{e}")
                    continue

            # 4. 循环结束，返回最终结果
            final_summary = "，".join(summary_logs)
            if not planned_actions:
                final_summary = "批量规划完成，但没有产生任何需要执行的后续动作。"
            return ToolResult(
                success=True,
                summary=final_summary,
                follow_up_action=planned_actions
            )
        
        except Exception as e:
            logger.error(f"批量规划时发生严重错误: {e}", exc_info=True)
            return ToolResult(success=False, summary="批量规划时发生严重错误。", error_message=str(e))