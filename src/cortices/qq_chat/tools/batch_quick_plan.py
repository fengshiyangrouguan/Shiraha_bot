import time
import uuid
import json
import asyncio
import re
from typing import Dict, Any, Optional, TYPE_CHECKING, List

from src.cortices.tools_base import BaseTool
from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter
from src.agent.world_model import WorldModel
from src.cortices.qq_chat.chat.qq_chat_data import QQChatData
from src.cortices.qq_chat.chat.chat_stream import QQChatStream
from src.common.event_model.event import Event
from src.common.event_model.event_data import Message, MessageSegment
from src.common.event_model.info_data import UserInfo
from src.common.event_model.info_data import ConversationInfo
from src.common.database.database_model import ConversationInfoDB, UserInfoDB
from src.llm_api.dto import LLMMessageBuilder
from src.llm_api.factory import LLMRequestFactory
from src.common.database.database_manager import DatabaseManager
from src.system.di.container import container
from src.common.logger import get_logger
from src.cortices.qq_chat.tools.utils.replyer import QQReplyer


logger = get_logger("qq_chat")

if TYPE_CHECKING:
    from src.cortices.qq_chat.cortex import QQChatCortex
    from src.cortices.manager import CortexManager


class BatchQuickPlanTool(BaseTool):
    """
    批量规划多个会话的行动意图。
    
    """
    def __init__(self, world_model: WorldModel, adapter: QQNapcatAdapter, cortex:"QQChatCortex",cortex_manager: "CortexManager", llm_request_factory: "LLMRequestFactory",database_manager: "DatabaseManager"):
        super().__init__(cortex_manager)
        self._world_model = world_model
        self.adapter = adapter
        self.cortex = cortex
        self.llm_request_factory = llm_request_factory
        self.database_manager = database_manager
        self.replyer = QQReplyer(world_model=self._world_model,adapter=self.adapter,llm_request_factory=self.llm_request_factory,database_manager=self.database_manager,cortex=self.cortex)
    @property
    def scope(self) -> str:
        return "qq_app"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "batch_quick_plan",
            "description": "批量规划多个会话的行动意图。请注意：若在历史行动中有和意图相似的行为，应在规划时进行去重，避免复读。",
            "parameters": {
                "type": "object",
                "properties": {
                    "replies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "conversation_id": {"type": "string", "description": "目标聊天 ID"},
                                "intent": {"type": "string", "description": "行动的核心意图"},
                            },
                        }
                    },
                }
            },
            "required": ["replies"]
        }
    
    
    def _build_quick_plan_prompt(self, conversation_info: ConversationInfo, intent: str, style: Dict[str, Any], history:str):
        """
        构造用于轻量级回复生成的 Prompt。
        """
        name = self._world_model.bot_name
        personality = self._world_model.bot_personality
        interest = self._world_model.bot_interest
        mood = self._world_model.mood
        # plan_style = self._world_model.bot_plan_style
        short_term_memory = "以下是按时间顺序排列的近期活动：\n"+"\n".join(self._world_model.short_term_memory) 

        if conversation_info.conversation_type == "group":
            chat_target = f"你正在群聊中与群友聊天。"
        else:
            chat_target = f"你正在与用户{conversation_info.conversation_name}进行私聊。"

        prompt = (
f"""
## 你的身份设定与当前状态:
- **你的名字**： {name}
- **你的性格**： {personality}
- **你的兴趣**： {interest}
- **你的当前情绪**： {mood}

{chat_target}

## **以下是最近聊天记录**,请仔细阅读：
{history}

## 待执行的社交规划
你当前的行动意图是："{intent}"

## 可用的 Action 规范
- **去重检查**：不要重复去执行你近期活动中已干过的事。
- **关联性**：若聊天内容与你无关，优先选择 no_reply。

1. **reply**: 进行快速回复然后关掉qq去干别的事情
{{
    "action": "reply",
    "reason": "回复的原因/意图"
}}

2. **deep_chat**: 想更深度长时间参与/观察某聊天，或需要在聊天中执行复杂任务，还可用于主动开启话题。
{{
    "action": "text",
    "reason": "深度聊天的原因/意图"
}}

3. **no_reply**: 保持沉默，不参与该会话。
{{
    "action": "no_reply",
    "reason": "沉默的理由"
}}


输出格式示例如下：
{{
    "action": "……",
    其他需要的参数……
}}

- **只输出 JSON 代码块**，不要任何多余文字。
请基于这些内容，选择一个action，生成JSON输出。
""")
        return prompt
    
    async def execute(self, replies: list[Dict[str, Any]]) -> str:
        """
        批量执行快速回复逻辑。
        replies: List of {conversation_id, intent, style}
        summary: 总结
        """
        results = []
        try:
            # 1. 获取 QQ 数据中心
            qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
            
            for item in replies:
                conversation_id = item.get("conversation_id")
                intent = item.get("intent")
                style = item.get("style", {})
                
                # 2. 获取会话上下文
                chat_stream = None
                if qq_chat_data and conversation_id in qq_chat_data.streams:
                    chat_stream = qq_chat_data.streams[conversation_id]
                    recent_messages = chat_stream.build_chat_history_has_msg_id()
                    conversation_info = chat_stream.conversation_info
                else:
                    # 尝试从数据库兜底
                    conv_db: ConversationInfoDB = await self.database_manager.get(ConversationInfoDB, conversation_id)
                    if not conv_db:
                        results.append({"id": conversation_id, "status": "skipped", "reason": "找不到会话记录"})
                        continue
                    
                    conversation_info = ConversationInfo(
                        conversation_id=conversation_id,
                        conversation_type=conv_db.conversation_type or "private",
                        conversation_name=conv_db.conversation_name or "未知对象"
                    )
                    recent_messages = "无最近聊天记录"

                # 3. 构造 Prompt 并请求 LLM 生成具体台词
                logger.info(f"我在{conversation_info.conversation_name}的聊天意图：{intent}")
                prompt = self._build_quick_plan_prompt(
                    conversation_info=conversation_info,
                    intent=intent,
                    style=style,
                    history=recent_messages
                )

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
                    
                    act_type = action.get("action")
                    reason = action.get("reason", "无理由")
                    logger.info(f"执行动作: {act_type}, 理由: {reason}")
                    if act_type == "no_reply":
                        logger.info(f"我决定不回复: {reason}")
                        results.append(f"在“{conversation_info.conversation_name}”聊天中，我决定不参与聊天，原因是{reason}")
                    elif act_type == "reply":
                        result = await self.replyer.execute(conversation_info,reason,chat_stream)
                        results.append(f"在“{conversation_info.conversation_name}”聊天中，{result}")
                        # 执行回复发送...
                    elif act_type == "deep_chat":
                        results.append(f"在“{conversation_info.conversation_name}”聊天中，你想开始深度聊天，但开发者还没做这个功能（哼哼你能拿我怎么样~）")

                    
                except Exception as e:
                    results.append(f"在“{conversation_info.conversation_name}”聊天中，行动解析失败了，报错:{e}")
                    continue

            result_str = "，".join(results)
            return result_str
        
        except Exception as e:
            return f"在规划多个会话的行动意图时发生崩溃: {e}"