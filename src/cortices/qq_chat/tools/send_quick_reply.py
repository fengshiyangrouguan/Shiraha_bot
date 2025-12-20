import time
import uuid
import json
import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING

from src.cortices.tools_base import BaseTool
from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter
from src.agent.world_model import WorldModel
from src.cortices.qq_chat.chat.qq_chat_data import QQChatData
from src.cortices.qq_chat.chat.chat_stream import QQChatStream
from src.common.event_model.event import Event
from src.common.event_model.event_data import Message, MessageSegment
from src.common.event_model.info_data import UserInfo
from src.llm_api.dto import LLMMessageBuilder
from src.system.di.container import container
from src.llm_api.factory import LLMRequestFactory

if TYPE_CHECKING:
    from src.cortices.qq_chat.cortex import QQChatCortex


class SendQuickReplyTool(BaseTool):
    """
    向指定的QQ聊天对象（用户或群组）发送一条简单的、一次性的消息。
    适用于不需要深入对话的场景。
    
    """
    def __init__(self, world_model: WorldModel, adapter: QQNapcatAdapter, cortex: "QQChatCortex"):
        self._world_model = world_model
        self.adapter = adapter
        self.cortex = cortex

    @property
    def scope(self) -> str:
        return "main"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "send_quick_qq_reply",
            "description": "向指定的QQ聊天对象（用户或群组）发送一条简单的消息。适用于不想或不需要深入聊天的场景。简单回应一下，回复的平淡一些，简短一些，不要描述动作，尽量少使用标点，一条回复可分几次发送，",
            "parameters": {
                "conversation_name": {
                    "type": "string",
                    "description": "目标聊天（私聊或群组）的名称。"
                },
                "intent":{
                    "type": "string",
                    "description": "来自 Planner 的高层回复意图，描述这次回复的意义、意图、目标。不是最终内容。"
                },
                "style": {
                    "type": "object",
                    "properties": {
                        "tone": {"type": "string","description": "回复的语气，例如：可爱、敷衍、平静、随意、正式、冷淡等。"},
                        "length": {"type": "string", "description": "句子长度偏好：短 / 中等 / 较长"},
                        "energy": {"type": "string", "description": "能量感：低 / 中 / 高"},
                        "role_manner": {"type": "string", "description": "人格表现方式：像朋友、像同事、像恋人、像孩子等"},
                    }
                },
            },
            "required_parameters": ["conversation_name", "intent", "style"]
        }
    
    async def _find_conversation_id(self, name: str) -> Optional[str]:
        """
        通过会话名称查找 conversation_id。
        来自 world_model 的内存（qq_chat_data）。
        """
        qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
        if not qq_chat_data:
            return None

        for cid, stream in qq_chat_data.streams.items():
            if stream.conversation_info.conversation_name == name:
                return cid

        return None
    
    def _build_quick_reply_prompt(self, chat_stream: QQChatStream, intent: str, style: Dict[str, Any], history:str):
        """
        构造用于轻量级回复生成的 Prompt。
        """
        name = self._world_model.bot_name
        personality = self._world_model.bot_personality
        interest = self._world_model.bot_interest
        mood = self._world_model.mood
        expression_style = self._world_model.bot_expression_style
        if chat_stream.conversation_info.conversation_type == "group":
          chat_target = f"你正在群聊{chat_stream.conversation_info.conversation_name}中与群友聊天。"
        else:
          chat_target = f"你正在与用户{chat_stream.conversation_info.conversation_name}进行私聊。"

        system_prompt = (
            "你是一个用于 QQ 聊天的『轻量聊天回复器』，负责生成自然、口语化、轻量的回应。\n"
            f"你的目标不是进行长对话，而是给出一两句自然的随口回复。\n"
            "\n"
            f"## 你的身份设定与当前状态:\n"
            f"- **你的名字**： {name}\n"
            f"- **你的性格**： {personality}\n"
            f"- **你的兴趣**： {interest}\n"
            f"- **你的当前情绪**： {mood}\n\n"  
            f"\n"
            f"## 回复任务设定：\n"
            f"- 回复核心意图：{intent}\n"
            f"- 说话风格：{expression_style}"
            f"- 语气: {style.get('tone', '平静')}\n"
            f"- 回复长度: {style.get('length', '短')}\n"
            f"- 回复能量感：{style.get('energy', '中')}\n"
            f"- 身份表现方式：{style.get('role_manner', '像朋友')}\n"
            f"\n"
            f"## 回复生成规范\n"
            f"- 不描述动作（例如不要写“我摇了摇头”等）\n"
            f"- 尽量少用标点\n"
            f"- 用自然、随口感的语气\n"
            f"- 回复风格不要太正式\n"
            f"- 可以分句，但不要太长\n"
            f"- 可以分成 1~3 条短句\n"
            f"- 不要复述“意图”本身\n"
            f"- 不要重复自己说的话"
            f"\n"
            f"## 输出格式\n"
            f"只输出 JSON，不要附加任何解释。\n"
            f"格式如下：\n"
            "{\n"
            "  \"segments\": [\"句1\", \"句2\"],   // 可以多条消息，按标点符号分割句子，每句不超过20字\n"
            "}\n"

        )
        user_prompt = (
            f"{chat_target}\n"
            f"以下是最近聊天记录：\n"
            f"{history}\n"
            f"请基于这些内容生成一个轻量、自然的回复。"
        )

        builder = LLMMessageBuilder()
        builder.add_system_message(system_prompt)
        builder.add_user_message(user_prompt)
        
        prompt = builder.get_message_dict()


        return prompt


    async def execute(self, conversation_name: str, intent: str, style: Dict[str, Any]) -> str:
        """
        执行发送快速回复的逻辑。
        """
        try:
            # 1. 从 WorldModel 获取上下文信息
            conversation_id = await self._find_conversation_id(conversation_name)
            if not conversation_id:
                return f"未找到名为 '{conversation_name}' 的会话，无法发送消息。"
            
            qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
            chat_stream = qq_chat_data.streams[conversation_id]
            recent_messages = chat_stream.build_chat_history_for_llm()
            conversation_info = chat_stream.conversation_info
            # 3. 构造用于 reply 的 prompt
            prompt = self._build_quick_reply_prompt(
                chat_stream=chat_stream,
                intent=intent,
                style=style,
                history=recent_messages
            )

            llm_factory = container.resolve(LLMRequestFactory)
            llm_request = llm_factory.get_request("replyer")
            content, model_name = await llm_request.execute(
                prompt=json.dumps(prompt)
            )
            try:
                reply_obj:Dict = json.loads(content)
                segments = reply_obj.get("segments", [])
                if not isinstance(segments, list):
                    raise ValueError("segments 不是列表")
            except Exception:
                # fallback：如果模型没按 JSON 返回，就把整段内容当成一句话
                segments = [content.strip()]

            # 若没有内容，不发送
            if len(segments) == 0:
                return "回复器 没有返回可发送内容"

            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue

                # 计算延时（打字模拟）: 0.15s / 字，范围 [0.6, 2.5]
                delay = min(max(len(seg) * 0.15, 0.6), 2.5)
                await asyncio.sleep(delay)

                # 发消息
                await self.adapter.message_api.send_text(conversation_info, seg)

                # ----------- 记录 event -----------
                bot_user_id = self.cortex.config.bot_id
                bot_user_info = UserInfo(
                    user_id=bot_user_id,
                    user_nickname=self._world_model.bot_name,
                    user_cardname=self._world_model.bot_name
                )
                message_seg = MessageSegment(
                    type="text",
                    data=seg
                )

                message_event_data = Message(
                    message_id=str(uuid.uuid4()),
                )

                message_event_data.add_segment(message_seg)

                new_event = Event(
                    event_type="message",
                    event_id=str(uuid.uuid4()),
                    time=int(time.time()),
                    platform=self.adapter.adapter_id,
                    chat_stream_id=conversation_id,
                    user_info=bot_user_info,
                    conversation_info=conversation_info,
                    event_data=message_event_data,
                )
                new_event.add_tag("self_message")

                await self.cortex.post_event_to_processor(new_event)

            return f"已针对回复意图({intent})完成回复"
        
        except Exception as e:
            return f"发送消息时出错: {e}"
