import asyncio
import json
import re
import time
import uuid
from typing import List, Optional

from src.agent.world_model import WorldModel
from src.common.database.database_manager import DatabaseManager
from src.common.di.container import container
from src.common.event_model.event import Event
from src.common.event_model.event_data import Message, MessageSegment
from src.common.event_model.info_data import ConversationInfo, UserInfo
from src.common.logger import get_logger
from src.cortices.qq_chat.chat.sticker_system.sticker_manager import StickerManager
from src.cortices.qq_chat.cortex import QQChatCortex
from src.cortices.qq_chat.data_model.chat_stream import QQChatStream
from src.llm_api.factory import LLMRequestFactory
from src.memory_system.services.expression_selector_service import ExpressionSelectorService
from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter
from src.platform.sources.qq_napcat.utils.image import file_path_to_base64

logger = get_logger("qq_quick_reply")


class QQReplyer:
    def __init__(
        self,
        world_model: WorldModel,
        adapter: QQNapcatAdapter,
        llm_request_factory: LLMRequestFactory,
        database_manager: DatabaseManager,
        cortex: QQChatCortex,
    ):
        self._world_model = world_model
        self.adapter = adapter
        self.llm_request_factory = llm_request_factory
        self.database_manager = database_manager
        self.cortex = cortex
        self.sticker_manager = container.resolve(StickerManager)
        self.expression_selector = container.resolve(ExpressionSelectorService)

    async def _build_reply_prompt(
        self,
        conversation_info: ConversationInfo,
        reason: str,
        history: list,
    ) -> list:
        name = self._world_model.bot_name
        personality = self._world_model.bot_personality
        interest = self._world_model.bot_interest
        mood = self._world_model.mood
        expression_style = self._world_model.bot_expression_style
        time = self._world_model.get_current_time_string()
        selected_expressions, selected_ids = await self.expression_selector.select_suitable_expressions(
            conversation_id=conversation_info.conversation_id,
            chat_info=history,
            max_num=5,
            reply_reason=reason,
        )
        logger.info(
            f"[QQReplyer] selected expression count={len(selected_expressions)}, "
            f"selected_ids={selected_ids}, conversation_id={conversation_info.conversation_id}"
        )

        expression_reference = self.expression_selector.format_selected_expressions_for_prompt(
            selected_expressions
        ) or "暂无可参考的历史表达方式。"

        short_term_memory = "\n".join(self._world_model.short_term_memory)
        if conversation_info.conversation_type == "group":
            chat_target = f"你正在群聊中与群友聊天。"
        else:
            chat_target = f"你正在与用户{conversation_info.conversation_name}进行私聊。"


        system_prompt =f"""
## 你的身份设定与当前状态:
- **你的名字**： {name}
- **你的性格**： {personality}
- **你的兴趣**： {interest}
- **你的当前情绪**： {mood}

- **你的近期活动**：
{short_term_memory}

{time}
{chat_target}

## **禁止回复 "—— 以上为已回复历史消息，禁止回复 ——" 上方的任何消息！！！**

- 你的说话风格：{expression_style}
### 你可以学习以下的说话表达风格，在合适的场景可直接套用句式：
{expression_reference}

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

{{ "action": "text
- **text**: 发消息", "content": "要发送的文本内容" }}

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
        
        messages = [{"role": "system", "content": system_prompt},]
        messages.extend(history)
        messages.append({"role": "user", "content": f"你现在想要回复消息，{reason}。请给出你的决策。"})
        logger.info(f"[QQReplyer] prompt built for conversation_id={conversation_info.conversation_id}")
        return messages
    async def _post_self_message_event(
        self,
        conversation_id: str,
        conversation_info: ConversationInfo,
        segs: List[MessageSegment],
    ) -> None:
        bot_user_info = UserInfo(
            user_id=self.cortex.config.bot_id,
            user_nickname=self._world_model.bot_name,
            user_cardname=self._world_model.bot_name,
        )
        message_event_data = Message(message_id=str(uuid.uuid4()))
        for seg in segs:
            message_event_data.add_segment(seg)

        new_event = Event(
            event_type="message",
            event_id=str(int(time.time() * 1000)),
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
        if chat_stream is None:
            return "未找到会话流。"

        try:
            conversation_info = chat_stream.conversation_info
            conversation_id = conversation_info.conversation_id
            recent_messages: list = chat_stream.build_openai_chat_history()
            prompt:list = await self._build_reply_prompt(conversation_info, reason, recent_messages)
            llm_request = self.llm_request_factory.get_request("replyer")
            content, _ = await llm_request.execute_messages(messages=prompt)
            logger.info(f"[QQReplyer] raw reply llm output: {content}")

            actions = self._parse_actions(content)
            logger.info(f"[QQReplyer] parsed actions: {actions}")
            if not actions:
                return "这次没有执行任何回复动作。"

            sent_any = False
            exit_reason = ""

            for act in actions:
                act_type = str(act.get("action", "")).strip()
                text_content = str(act.get("content", "")).strip()

                if act_type == "exit":
                    exit_reason = str(act.get("reason", "")).strip()
                    logger.info(f"[QQReplyer] received exit action: reason={exit_reason}")
                    continue

                try:
                    if act_type == "reply":
                        msg_id = act.get("message_id")
                        logger.info(
                            f"[QQReplyer] sending reply action: message_id={msg_id}, content={text_content}"
                        )
                        if msg_id:
                            await self.adapter.message_api.send_text(
                                conversation_info=conversation_info,
                                text=text_content,
                                reply_id=msg_id,
                            )
                            segs = [
                                MessageSegment(type="reply", data=msg_id),
                                MessageSegment(type="text", data=text_content),
                            ]
                        else:
                            await self.adapter.message_api.send_text(conversation_info, text_content)
                            segs = [MessageSegment(type="text", data=text_content)]
                        await self._post_self_message_event(conversation_id, conversation_info, segs)
                        sent_any = True

                    elif act_type == "text" and text_content:
                        logger.info(f"[QQReplyer] sending text action: content={text_content}")
                        segments = [seg.strip() for seg in re.split(r"[,，\s]+", text_content) if seg.strip()]
                        await self._send_text_with_segments(conversation_info, segments)
                        sent_any = True

                    elif act_type == "sticker":
                        sticker_emotion = str(act.get("sticker_emotion", "")).strip()
                        logger.info(f"[QQReplyer] sending sticker action: emotion={sticker_emotion}")
                        if sticker_emotion:
                            llm_embedding_request = self.llm_request_factory.get_request("embedding")
                            embedding, _ = await llm_embedding_request.execute_embedding(sticker_emotion)
                            file_path = self.sticker_manager.search_stickers(embedding)
                            logger.info(f"[QQReplyer] sticker search result: {file_path}")
                            if file_path:
                                await self.adapter.message_api.send_sticker(conversation_info, file_path)
                                file_base64 = file_path_to_base64(file_path)
                                seg = MessageSegment(type="sticker", data=file_base64)
                                await self._post_self_message_event(conversation_id, conversation_info, [seg])
                                sent_any = True
                except Exception as exc:
                    logger.warning(f"[QQReplyer] action execution failed: action={act_type}, error={exc}")
                    continue

            logger.info(f"[QQReplyer] sent_any={sent_any}, exit_reason={exit_reason}")
            if sent_any:
                chat_stream.mark_as_replyed()
                await asyncio.sleep(30)
                summary = await self._summary_action(chat_stream, reason)
                logger.info(f"[QQReplyer] summary after send: {summary}")
                return summary

            if exit_reason:
                return f"这次没有继续回复，因为{exit_reason}"
            return "这次没有实际发出回复。"
        except Exception as exc:
            logger.error(f"[QQReplyer] execute failed: {exc}", exc_info=True)
            return f"执行回复时发生错误: {exc}"

    async def _send_text_with_segments(self, conversation_info: ConversationInfo, segments: List[str]) -> None:
        for text in segments:
            delay = min(max(len(text) * 0.15, 0.6), 2.5)
            await asyncio.sleep(delay)
            await self.adapter.message_api.send_text(conversation_info, text)
            message_seg = MessageSegment(type="text", data=text)
            await self._post_self_message_event(conversation_info.conversation_id, conversation_info, [message_seg])

    async def _summary_action(self, chat_stream: QQChatStream, reason: str) -> str:
        recent_messages = chat_stream.build_chat_history_for_summary()
        llm_request = self.llm_request_factory.get_request("utils_small")
        prompt = f"""
请你总结这次轻量回复行为，输出一小段自然语言摘要。

这次回复原因：
{reason}

会话记录：
{recent_messages}

要求：
- 总结是否真正完成了互动。
- 如果有发送消息，概括回复了什么。
- 如果没有发送，也要明确说明。
- 控制在 80 字内。
"""
        content, _ = await llm_request.execute(prompt=prompt)
        return content

    @staticmethod
    def _parse_actions(content: str) -> List[dict]:
        json_str = content.strip()
        if json_str.startswith("```"):
            json_str = json_str.replace("```json", "").replace("```", "").strip()
        try:
            payload = json.loads(json_str)
        except Exception:
            logger.warning(f"[QQReplyer] reply output is not valid JSON: {content}")
            return []

        actions = payload.get("actions", [])
        return actions if isinstance(actions, list) else []
