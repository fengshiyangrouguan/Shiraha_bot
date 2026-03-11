import time
import uuid
import json
import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING, List

from src.cortices.tools_base import BaseTool
from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter
from src.agent.world_model import WorldModel
from src.cortices.qq_chat.data_model.qq_chat_data import QQChatData
from src.common.event_model.info_data import ConversationInfo
from src.llm_api.factory import LLMRequestFactory
from src.common.database.database_manager import DatabaseManager
from src.cortices.qq_chat.chat.replyer import QQReplyer
from src.cortices.qq_chat.cortex import QQChatCortex
from src.common.logger import get_logger


logger = get_logger("qq_deep_chat")

if TYPE_CHECKING:
    from src.cortices.manager import CortexManager

class DeepChatPlanner():
    """
    深度对话工具：
    用于处理需要长期交流的会话。
    """
    def __init__(self, world_model: WorldModel, adapter: QQNapcatAdapter, llm_request_factory: LLMRequestFactory, database_manager:DatabaseManager, cortex:QQChatCortex, replyer:QQReplyer):
        self._world_model = world_model
        self.adapter = adapter
        self.llm_request_factory = llm_request_factory
        self.database_manager = database_manager
        self.cortex = cortex
        self.replyer = replyer
    def _build_prompt(self, conversation_info: ConversationInfo, history: str, intent: str, act_result: str, loop_len: int):
        """构造高质量深度回复 Prompt"""
        name = self._world_model.bot_name
        personality = self._world_model.bot_personality
        mood = self._world_model.mood
        interest = self._world_model.bot_interest
        if loop_len == 1:
            init_intent = f"""## 待执行的社交规划
你当前的行动意图是："{intent}"
"""        
        else:
            init_intent = ""
            
        if conversation_info.conversation_type == "group":
            chat_target = f"你正在群聊中与群友聊天。"
        else:
            chat_target = f"你正在与用户{conversation_info.conversation_name}进行私聊。"

        return f"""
## 你的身份设定与当前状态:
- **你的名字**： {name}
- **你的性格**： {personality}
- **你的兴趣**： {interest}
- **你的当前情绪**： {mood}

{chat_target}

## **你之前的行动记录**：
{act_result}

## **以下是最近聊天记录**,请仔细阅读：
{history}

{init_intent}

## 可用的 Action 规范
- **去重检查**：不要重复去执行你近期活动中已干过的事。
**注意！“我自己”指的就是你自己发过的消息，请不要回复你自己的消息**
- **关联性**：若无人回复你，聊天内容与你无关，或聊天比较冷清，优先选择 wait_and_see。

1. **reply**: 发送/回复消息
{{
    "action": "reply",
    "reason": "发送消息的原因/意图"
}}

2. **wait_and_see**: 保持沉默，持续观察聊天
{{
    "action": "wait_and_see",
    "reason": "沉默的理由"
}}

3. **exit**: 退出深度聊天(例如不想聊了，或动机已完成)
{{
    "action": "exit",
    "reason": "退出的理由"
}}


输出格式示例如下：
{{
    "action": "……",
    其他需要的参数……
}}

- **只输出 JSON 代码块**，不要任何多余文字。
请基于这些内容，选择一个action，生成JSON输出。
"""

    async def enter_deep_chat(self, conversation_info: ConversationInfo, intent: str) -> str:
        logger.info(f"进入深度对话模式 -> {conversation_info.conversation_name}，初始意图: {intent}")
        max_loop_len = 15
        loop_len = 0
        results = ["无"]
        while loop_len < max_loop_len:
            loop_len += 1
            try:
                # 1. 获取会话数据
                qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
                conversation_id = conversation_info.conversation_id

                chat_stream = None
                if qq_chat_data and conversation_id in qq_chat_data.streams:
                    chat_stream = qq_chat_data.streams[conversation_id]
                    history = chat_stream.build_chat_history_has_msg_id()
                else:
                    history = "无最近聊天记录"

                # 2. 构造 Prompt
                act_result = "\n".join(results)
                prompt = self._build_prompt(conversation_info, history, intent,act_result, loop_len)

                llm_request = self.llm_request_factory.get_request("planner") 
                content, _ = await llm_request.execute(prompt=prompt)

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

                    if act_type == "wait_and_see":
                        logger.info(f"我决定不回复: {reason}")
                        await asyncio.sleep(30)
                        result = f"我决定沉默并观察聊天，原因是{reason}"
                        results.append(result)

                    elif act_type == "reply":
                        result = await self.replyer.execute(conversation_info,reason,chat_stream)
                        results.append(result)

                    elif act_type == "exit":
                        result = f"我决定退出聊天，原因是{reason}"
                        results.append(result)
                        deep_chat_result = await self._summary_action(intent, act_result)
                        return deep_chat_result
                    
                    result = f"我决定退出聊天，我聊天太久了，感觉需要去干点别的了"
                    deep_chat_result = await self._summary_action(intent, act_result)
                    return deep_chat_result

                except Exception as e:
                    result=f"行动解析失败了，报错:{e}"
                    continue


            except Exception as e:
                logger.error(f"DeepChat 执行崩溃: {e}")
                return f"深度对话执行失败: {e}"
            
    async def _summary_action(self, intent, act_result):

        llm_request = self.llm_request_factory.get_request("utils_small")
        prompt = f"""
你是一个行动总结器。请将行动记录压缩总结为自我行为记录。

## 原始执行数据
- 你由于"{intent}"的初始意图，开启了深度的聊天
- 行动记录：
{act_result}

## 归纳要求
1. **第一人称**：必须以“我...”开头。
2. 要求流畅地描述我因为什么意图干了什么
2. 不超过 50 字。
3. **情感一致性**：根据发送的内容，推测并保留自己当时的情绪色彩。
"""
        content, _ = await llm_request.execute(prompt=prompt)
        return content