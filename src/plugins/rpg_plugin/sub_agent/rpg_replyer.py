import json
import asyncio
import re
import uuid
import time
from typing import Dict, Any, List, Optional
from src.common.logger import get_logger
from src.agent.world_model import WorldModel
from src.common.event_model.event import Event
from src.cortices.qq_chat.chat.sticker_system.sticker_manager import StickerManager
from src.cortices.qq_chat.api import QQChatAPI
from src.cortices.qq_chat.data_model.chat_stream import QQChatStream, QQChatMessage
from src.common.event_model.event_data import Message, MessageSegment
from src.common.event_model.info_data import UserInfo
from src.common.event_model.info_data import ConversationInfo
from src.cortices.manager import CortexManager

from src.common.di.container import container
from src.llm_api.factory import LLMRequestFactory
from src.common.database.database_manager import DatabaseManager

logger = get_logger("rpg_replyer")
    
class RPGReplyer:
    """
    RPG 专用回复器：封装了身份切换、消息发送、表情包匹配及行为总结。
    """
    def __init__(self, chat_stream: QQChatStream):
        self.world_model = container.resolve(WorldModel)
        self.llm_factory = container.resolve(LLMRequestFactory)
        self.sticker_manager = container.resolve(StickerManager)
        self.message_api = container.resolve(QQChatAPI)
        self.chat_stream: QQChatStream = chat_stream
        
        # 内部共用的 LLM 请求实例
        self.reply_llm = self.llm_factory.get_request("replyer")
        self.summary_llm = self.llm_factory.get_request("utils_small")

    def _build_prompt(self, identity_config: Dict, intent: str, history: str, message:Optional[QQChatMessage] = None) -> str:
        """动态组织提示词"""
        name = self.world_model.bot_name
        personality = self.world_model.bot_personality
        expression_style = self.world_model.bot_expression_style
        sticker_tool = ""
        attention_prompt = ""
        if message:
            attention_prompt = f"现在，{message.user_nickname}的消息 {message.content} 吸引了你的注意，请你进行回复。"
        else:
            attention_prompt = "现在，请根据当前场景主动发言。"
        prompt = f"""
## 你的人物设定:
- **你的名字**： {name}
- **你的性格**： {personality}
- **说话风格**：{expression_style}

- **当前扮演的身份**: {identity_config.get('identity')}
- **需要表现的风格**: {identity_config.get('style')}
- **身份任务**: {identity_config.get('task')}

## 场景上下文:
- **当前目标**: {intent}
- **行动记忆**: 
{"\n".join(self.world_model.short_term_memory) }

## 聊天记录:
{history}

## 回复规范:
2. 回复限制：{identity_config.get('action_limit', '禁止描述动作')}。

{attention_prompt}


你可以按情况随意选择以一个或多个action:
**action可重复使用多次**

输出格式示例如下：
{{
    "actions": [
        {{"content": "要发送的文本内容"}},
        ……
    ]   
}}

请基于这些内容生成JSON输出。

"""
        return prompt

    async def execute_performance(self, identity_config: Dict, intent: str, message: Optional[QQChatMessage] = None):
        """
        外界调用的唯一入口。
        identity_config: 包含 name, style, task 等的字典
        reason: 为什么要发这条消息
        """
        self.identity_config = identity_config
        history = self.chat_stream.build_chat_history_has_msg_id()
        prompt = self._build_prompt(identity_config, intent, history,message )
        
        # 1. 获取 LLM 决策
        content, _ = await self.reply_llm.execute(prompt=prompt)
        actions = self._parse_json(content)
        
        # 2. 执行具体动作 (复用你的分段发送和表情包逻辑)
        reply_id = None
        if message:
            reply_id = message.message_id
        for act in actions:
            await self.message_api.send_text(conversation_info=self.chat_stream.conversation_info, text=act["content"],reply_id=reply_id)
            typing_delay = min(len(act["content"]) * 0.15 + 0.5, 5.0)
            await asyncio.sleep(typing_delay)
        return

    def _parse_json(self, content: str) -> List[Dict]:
        """安全解析 JSON"""
        try:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            return json.loads(match.group()).get("actions", []) if match else []
        except: return []

    async def _summary_self_action(self, reason, history):
        """总结本次演出的行为"""
        # 调用 summary_llm 归纳...
        pass

    async def _post_self_message_event(self, conversation_id: str, conversation_info: ConversationInfo, segs:List[MessageSegment]):
        """内部方法：将发送的消息包装成事件发回处理器"""
        cortex_manager = container.resolve(CortexManager)
        cortex = cortex_manager._cortices["QQ Chat Cortex"]
        bot_user_info = UserInfo(
            user_id= "352699068",
            user_nickname=self.world_model.bot_name,
            user_cardname=self.world_model.bot_name
        )
        message_event_data = Message(message_id=str(uuid.uuid4()))
        for message_seg in segs:
            message_event_data.add_segment(message_seg)

        new_event = Event(
            event_type="message",
            event_id=str(int(time.time())),
            time=int(time.time()),
            platform="qq_chat_adapter",
            chat_stream_id=conversation_id,
            user_info=bot_user_info,
            conversation_info=conversation_info,
            event_data=message_event_data,
        )
        new_event.add_tag("self_message")
        await cortex.post_event_to_processor(new_event)