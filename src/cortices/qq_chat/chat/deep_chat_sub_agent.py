import time
import uuid
import json
import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING, List

from src.cortices.tools_base import BaseTool
from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter
from src.agent.world_model import WorldModel
from src.cortices.qq_chat.data_model.chat_stream import QQChatStream
from src.common.event_model.info_data import ConversationInfo
from src.llm_api.factory import LLMRequestFactory
from src.common.database.database_manager import DatabaseManager
from src.cortices.qq_chat.chat.replyer import QQReplyer
from src.cortices.qq_chat.cortex import QQChatCortex
from src.common.logger import get_logger
from src.common.di.container import container


logger = get_logger("qq_deep_chat")

if TYPE_CHECKING:
    from src.cortices.manager import CortexManager

class DeepChatSubAgent():
    """
    深度聊天子智能体。
    负责在单个会话中进行持续的、有状态的交互。
    它拥有自己的“感知-规划-行动”循环。
    """
    def __init__(self, adapter: QQNapcatAdapter, cortex:QQChatCortex, replyer:QQReplyer): 
        self.adapter = adapter
        self.cortex = cortex
        self.replyer = replyer
        self._world_model: WorldModel = container.resolve(WorldModel)
        self.llm_request_factory: LLMRequestFactory = container.resolve(LLMRequestFactory)
        self.database_manager:DatabaseManager = container.resolve(DatabaseManager)

    def _build_prompt(self, conversation_info: ConversationInfo, history: str, intent: str, act_result: str, loop_len: int):
        """构造深度回复 Prompt"""
        name = self._world_model.bot_name
        personality = self._world_model.bot_personality
        mood = self._world_model.mood
        interest = self._world_model.bot_interest
        time = self._world_model.get_current_time_string()
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

{time}
{chat_target}

## **你之前的行动记录**：
{act_result}

## **以下是最近聊天记录**,请仔细阅读：
{history}

{init_intent}

## 可用的 Action 规范
- **去重检查**：不要重复去执行你近期活动中已干过的事。
**注意！“你自己”指的就是你自己发过的消息，请不要回复你自己的消息**

**禁止回复 "—— 以上为已回复历史消息，禁止回复 ——" 上方的任何消息！！！**

不要对表情包进行回复
- **关联性**：若无人回复你，聊天内容与你无关，或聊天比较冷清，可以选择 wait_for_message 持续观察
- **主动退出**：如果观察时间够长了，或者要去干别的事情 选择 exit 来退出聊天


1. **reply**: 发送/回复消息
{{
    "action": "reply",
    "parameters":{
        "reason": "发送消息的原因/意图"
    }
}}

2. **wait_for_message**: 保持沉默，持续观察聊天
{{
    "action": "wait_for_message",
    "parameters":{
        "reason": "沉默的理由"
    }
}}

3. **exit**: 退出深度聊天
{{
    "action": "exit",
    "parameters":{
        "reason": "退出的理由"
        "follow_action":{
            "action":"退出深度聊天后想要执行的行动名称，如果没有则输入“None”"
            "parameters":"调用的工具需要的参数"
        }
    }
}}


输出格式示例如下：
{{
    "action": "行动名称",
    "parameters":"需要的参数"
}}

- **只输出 JSON 代码块**，不要任何多余文字。
请基于这些内容，选择一个action，生成JSON输出。
"""

    async def run(self, intent: str, chat_stream: Optional[QQChatStream]= None) -> str:
        conversation_info:ConversationInfo = chat_stream.conversation_info
        logger.info(f"子智能体启动：进入深度对话模式 -> {conversation_info.conversation_name}，初始意图: {intent}")
        max_loop_len = 15
        loop_len = 0
        results = ["无"]
        should_plan_immediately = True
        chat_stream = chat_stream
        while loop_len < max_loop_len:
            loop_len += 1            
            try:
                history = chat_stream.build_chat_history_has_msg_id
                new_message_event = chat_stream.get_new_message_event()
                if not should_plan_immediately:
                    try:
                        await asyncio.wait_for(new_message_event.wait(), timeout=180)
                        new_message_event.clear()
                        logger.info(f"[{conversation_info.conversation_name}] 检测到新消息，继续开始规划。")
                    except asyncio.TimeoutError:
                        result = f"等了好久都没有新消息，感觉需要离开去干点别的了"
                        results.append(result)
                        should_plan_immediately = True

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

                    if act_type == "wait_for_message":
                        logger.info(f"我决定不回复: {reason}")
                        result = f"我决定沉默并观察聊天，原因是{reason}"
                        should_plan_immediately = False
                        results.append(result)

                    elif act_type == "reply":
                        result = await self.replyer.execute(reason,chat_stream)
                        results.append(result)

                    elif act_type == "exit":
                        result = f"我决定退出聊天，原因是{reason}"
                        results.append(result)
                        deep_chat_result = await self._summary_action(intent, act_result,history)
                        return deep_chat_result
            
                except Exception as e:
                    result=f"行动解析失败了，报错:{e}"
                    continue


            except Exception as e:
                logger.error(f"DeepChat 执行崩溃: {e}")
                return f"深度对话执行失败: {e}"
        
        result = f"我决定退出聊天，我聊天太久了，感觉需要去干点别的了"
        deep_chat_result = await self._summary_action(intent, act_result)
        return deep_chat_result
            
    async def _summary_action(self, intent, act_result, history):

        llm_request = self.llm_request_factory.get_request("planner")
        prompt = f"""
你是一个行动总结器。请将行动记录压缩总结为自我行为记录。

## 原始执行数据
- 你由于"{intent}"的初始意图，开启了深度的聊天
- 行动记录：
{act_result}

-聊天记录：
{history}

## 归纳要求
1. **第一人称**：必须以“我...”开头。
2. 要求流畅地描述我因为什么意图干了什么,最终结果是什么
2. 50 字左右。
3. **情感一致性**：根据发送的内容，推测并保留自己当时的情绪色彩。
"""
        content, _ = await llm_request.execute(prompt=prompt)
        return content