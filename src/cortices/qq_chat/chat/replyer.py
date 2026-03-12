import time
import uuid
import json
import asyncio
import re
from typing import Dict, Any, Optional, TYPE_CHECKING, List

from src.cortices.tools_base import BaseTool
from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter
from src.agent.world_model import WorldModel
from src.cortices.qq_chat.data_model.qq_chat_data import QQChatData
from src.cortices.qq_chat.data_model.chat_stream import QQChatStream
from src.common.event_model.event import Event
from src.common.event_model.event_data import Message, MessageSegment
from src.common.event_model.info_data import UserInfo
from src.common.event_model.info_data import ConversationInfo
from src.common.database.database_model import ConversationInfoDB, UserInfoDB
from src.llm_api.dto import LLMMessageBuilder
from src.llm_api.factory import LLMRequestFactory
from src.common.database.database_manager import DatabaseManager
from src.cortices.qq_chat.chat.sticker_system.sticker_manager import StickerManager
from src.platform.sources.qq_napcat.utils.image import file_path_to_base64
from src.common.di.container import container
from src.common.logger import get_logger
from src.cortices.qq_chat.cortex import QQChatCortex

logger = get_logger("qq_quick_reply")
    


class QQReplyer:
    """
    渲染层执行器 (Renderer)：
    将规划层的意图（Intent）转化为具体的 QQ 动作指令（Action）。
    """

    def __init__(self, world_model: WorldModel, adapter: QQNapcatAdapter, llm_request_factory: LLMRequestFactory, database_manager:DatabaseManager, cortex:QQChatCortex):
        self._world_model = world_model
        self.adapter = adapter
        self.llm_request_factory = llm_request_factory
        self.database_manager = database_manager
        self.cortex = cortex
        self.sticker_manager = container.resolve(StickerManager)

    def _build_reply_prompt(self, conversation_info: ConversationInfo, reason: str, history:str):
        """
        构造用于轻量级回复生成的 Prompt。
        """
        name = self._world_model.bot_name
        personality = self._world_model.bot_personality
        interest = self._world_model.bot_interest
        mood = self._world_model.mood
        expression_style = self._world_model.bot_expression_style
        short_term_memory = "以下是按时间顺序排列的近期活动：\n"+"\n".join(self._world_model.short_term_memory) 

        if conversation_info.conversation_type == "group":
            chat_target = f"你正在群聊中与群友聊天。"
        else:
            chat_target = f"你正在与用户{conversation_info.conversation_name}进行私聊。"

        prompt = f"""
## 你的身份设定与当前状态:
- **你的名字**： {name}
- **你的性格**： {personality}
- **你的兴趣**： {interest}
- **你的当前情绪**： {mood}

- **你的近期活动**：
{short_term_memory}

{chat_target}

## **以下是最近聊天记录**,请仔细阅读：
{history}

## **注意！“你自己”指的就是你自己发过的消息，请不要回应你自己的消息**
## **禁止回复 "—— 以上为已回复历史消息，禁止回复 ——" 上方的任何消息！！！**

你现在想要回复消息，{reason}
- 说话风格：{expression_style}

现在请你读读之前的聊天记录，然后给出日常且口语化的回复，平淡一些，尽量简短一些。
- 不描述动作（例如不要写“我摇了摇头”等）
- 尽量使用短句
- 最终结尾处禁止使用句号
- 不要对表情包进行回复



## 输出格式
你可以按情况随意选择以下一个或多个action:
**action可重复使用多次，例如使用多次text，使用多次sticker**

## 行动规则：
**注意自己的发消息频率，适当控制**

- **reply**: 回复某一句话
{{ "action": "reply", "message_id": "要回复的消息的消息ID", "content": "要回复的简短文本内容" }}

- **sticker**: 发表情包
{{ "action": "sticker", "sticker_emotion": "想表达的情感或内容" }}

- **text**: 发消息
{{ "action": "text", "content": "要发送的文本内容" }}

- **exit**: 这次不发消息了，当你觉得现在不适宜发消息请只选择输出这一个action
{{ "action": "exit", "reason": "不参与聊天的原因" }}

输出格式示例如下：
{{
    "actions": [
        {{ "action": "……" }},
        ……
    ]   
}}

请基于这些内容生成JSON输出。
"""
        return prompt

    async def _post_self_message_event(self, conversation_id: str, conversation_info: ConversationInfo, segs:List[MessageSegment]):
        """内部方法：将发送的消息包装成事件发回处理器"""
        bot_user_info = UserInfo(
            user_id=self.cortex.config.bot_id,
            user_nickname=self._world_model.bot_name,
            user_cardname=self._world_model.bot_name
        )
        message_event_data = Message(message_id=str(uuid.uuid4()))
        for message_seg in segs:
            message_event_data.add_segment(message_seg)

        new_event = Event(
            event_type="message",
            event_id=str(int(time.time())),
            time=int(time.time()),
            platform=self.adapter.adapter_id,
            chat_stream_id=conversation_id,
            user_info=bot_user_info,
            conversation_info=conversation_info,
            event_data=message_event_data,
        )
        new_event.add_tag("self_message")
        await self.cortex.post_event_to_processor(new_event)

    async def execute(self, reason: str, chat_stream: Optional[QQChatStream] = None) -> str:
        """
        批量执行快速回复逻辑。
        """
        result = []
        try:
            chat_stream = chat_stream
            conversation_info = chat_stream.conversation_info
            conversation_id = conversation_info.conversation_id
            
            recent_messages = chat_stream.build_chat_history_has_msg_id()


            # 3. 构造 Prompt 并请求 LLM 生成具体台词
            logger.info(f"我在{conversation_info.conversation_name}的聊天意图：{reason}")
            logger.info(recent_messages)
            prompt = self._build_reply_prompt(
                conversation_info=conversation_info,
                reason=reason,
                history=recent_messages
            )

            llm_factory = self.llm_request_factory
            llm_request = llm_factory.get_request("replyer")
            content, _ = await llm_request.execute(prompt=prompt)
            
            # 4. 解析生成的JSON
            try:
                json_str = content.strip()
                if json_str.startswith("```"):
                    json_str = json_str.replace("```json", "").replace("```", "").strip()
                reply_obj = json.loads(json_str)
                actions = reply_obj.get("actions", [])
            except Exception as e:
                result = f"我想回复，但出错了：{e}"

            if not actions:
                result = "我临时决定不回复"
            
            # 5. 执行具体 Action
            for act in actions:
                act_type = act.get("action")
                text_content = act.get("content", "").strip()
                try:
                    if act_type == "exit":
                        break
                    if act_type == "reply":
                        # 引用回复
                        msg_id = act.get("message_id")
                        if msg_id:
                            await self.adapter.message_api.send_text(
                                conversation_info=conversation_info, 
                                text=text_content, 
                                reply_id=msg_id)
                        else:
                            # 降级为普通文本
                            await self.adapter.message_api.send_text(conversation_info, text_content)
                        seg_reply = MessageSegment(type="reply",data=msg_id)
                        seg_text = MessageSegment(type="text",data=text_content)
                        segs=[seg_reply,seg_text]
                        await self._post_self_message_event(conversation_id, conversation_info, segs)

                    elif act_type == "at":
                        # @ 某人
                        at_person = act.get("at_person", "")

                    elif act_type == "text":
                        # 普通文本
                        if text_content:
                            segments = re.split(r'[,，\s]+', text_content)
                            await self._send_text_with_segments(conversation_info, segments)

                    elif act_type == "sticker":
                        sticker_emotion = act.get("sticker_emotion", "")
                        if sticker_emotion:
                            llm_embedding_request = llm_factory.get_request("embedding")
                            embedding, _ = await llm_embedding_request.execute_embedding(sticker_emotion)
                            logger.info(f"根据情感描述 '{sticker_emotion}'，开始查找")  # 打印前5维作为示例
                            file_path = self.sticker_manager.search_stickers(embedding)
                            logger.info(f"查找完成，结果文件路径: {file_path}")
                            if file_path:
                                logger.info(f"找到的表情包路径: {file_path}")
                                await self.adapter.message_api.send_sticker(conversation_info, file_path)
                                logger.info(f"发送表情包")
                                file_base64 = file_path_to_base64(file_path)
                                seg = MessageSegment(type="sticker",data=file_base64)
                                await self._post_self_message_event(conversation_info.conversation_id, conversation_info, [seg])
                            else:
                                logger.info(f"未匹配成功")                
                except Exception as e:
                    continue
            # 记录 Event 到系统（用于更新短期记忆和上下文） 
            # 观察一下发送消息的反应
            chat_stream.mark_as_replyed()
            await asyncio.sleep(10)
            result = await self._summary_action(chat_stream,reason)
            return result
            
        except Exception as e:
            return f"我回复失败，因为: {e}"

    async def _send_text_with_segments(self, conversation_info: ConversationInfo, segments: list[str]):
        """辅助方法：将文本分段发送，并在每段之间模拟打字延迟"""
        for text in segments:
            text = text.strip()
            if not text: continue
            
            # 模拟打字延迟
            delay = min(max(len(text) * 0.15, 0.6), 2.5)
            await asyncio.sleep(delay)
            
            # 实际发送
            await self.adapter.message_api.send_text(conversation_info, text)
            message_seg = MessageSegment(type="text",data=text)
            await self._post_self_message_event(conversation_info.conversation_id, conversation_info, [message_seg])

    async def _summary_action(self, chat_stream: QQChatStream, reason):

        recent_messages = chat_stream.build_chat_history_for_summary()
        llm_request = self.llm_request_factory.get_request("utils_small")
        prompt = f"""
你是一个行动总结器。请将意图和聊天记录压缩总结为自我行为记录。

## 原始执行数据
- 意图（之前想干什么）："{reason}"
- 聊天记录（“你自己”指的就是你本人）
{recent_messages}

## 归纳要求
1. **第一人称**：必须以“我...”开头。
2. 要求流畅地描述我因为什么意图干了什么，聊天中除了我自己其他人有没有反应，反应是什么，判断发的消息是否太多了，**如果没有反应请明确说明**
2. 输出50 字左右。
3. **情感一致性**：根据发送的内容，推测并保留自己当时的情绪色彩。

"""
        content, _ = await llm_request.execute(prompt=prompt)
        return content