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
from src.cortices.qq_chat.chat.sticker_system.sticker_manager import StickerManager
from src.platform.sources.qq_napcat.utils.image import file_path_to_base64
from src.system.di.container import container
from src.common.logger import get_logger

logger = get_logger("qq_quick_reply")

if TYPE_CHECKING:
    from src.cortices.qq_chat.cortex import QQChatCortex
    from src.cortices.manager import CortexManager


class SendQuickReplyTool(BaseTool):
    """
    向指定的QQ聊天对象（用户或群组）发送一条简单的、一次性的消息。
    适用于不需要深入对话的场景。
    
    """
    def __init__(self, world_model: WorldModel, adapter: QQNapcatAdapter, cortex: "QQChatCortex", cortex_manager: "CortexManager", llm_request_factory: "LLMRequestFactory",database_manager: "DatabaseManager"):
        super().__init__(cortex_manager)
        self._world_model = world_model
        self.adapter = adapter
        self.cortex = cortex
        self.llm_request_factory = llm_request_factory
        self.database_manager = database_manager
        self.sticker_manager = container.resolve(StickerManager)

    @property
    def scope(self) -> str:
        return "qq_app"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "batch_quick_reply",
            "description": "批量简单回复一下多个聊天会话。为每个会话产生不同的语气、长度和人格表现。你可以选择性地回复一些会话，不需要每个都回复。也可以按情况随意搭配回复风格，就算有些时候略显矛盾",
            "parameters": {
                "type": "object",
                "properties": {
                    "replies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "conversation_id": {"type": "string", "description": "目标 ID"},
                                "intent": {"type": "string", "description": "回复的核心意图"},
                                "style": {
                                    "type": "object",
                                    "properties": {
                                        "tone": {"type": "string", "description": "语气：敷衍/热情/冷淡/可爱等"},
                                        "energy": {"type": "string", "description": "能量感：低/中/高"},
                                        "role_manner": {"type": "string", "description": "关系定位：像朋友/像陌生人等"}
                                    } 
                                }
                            },
                        }
                    },
                    "summary": {"type": "string", "description": "对于本次活动的总结,以第一人称“我”感性地概括一下这次回复在具体哪些群聊或私聊中都分布都干了什么，有什么想法"}
                }
            },
            "required": ["replies", "summary"]
        }
    
    
    def _build_quick_reply_prompt(self, conversation_info: ConversationInfo, intent: str, style: Dict[str, Any], history:str):
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
            # group_action = (
            #     """      
            #         at: @某个人，用于引起某人注意（通常是群聊中的人）:
            #         {action: "at",at_name: "这是你要@的人的名字，通常是群聊中的人",content(at时非必要): "要发送的文本内容"}
            #     """
            # )
            # chat_target = f"你正在群聊{chat_stream.conversation_info.conversation_name}中与群友聊天。"
        else:
            chat_target = f"你正在与用户{conversation_info.conversation_name}进行私聊。"

        prompt = (
            "你是一个具有复杂情感的 QQ 聊天回复器，负责生成自然、口语化、轻量的回应。\n"
            f"你的目标不是进行长对话，而是给出一两句自然的随口回复。\n"
            "\n"
            f"## 你的身份设定与当前状态:\n"
            f"- **你的名字**： {name}\n"
            f"- **你的性格**： {personality}\n"
            f"- **你的兴趣**： {interest}\n"
            f"- **你的当前情绪**： {mood}\n\n"  
            f"- **你的近期活动**：\n{short_term_memory}\n"
            f"\n"
            f"{chat_target}\n"
            f"## **以下是最近聊天记录**,请仔细阅读：\n"
            f"{history}\n\n"
            f"## 回复任务设定：\n"
            f"- 回复核心意图：{intent}\n"
            f"- 说话风格：{expression_style}"
            f"- 语气: {style.get('tone', '平静')}\n"
            f"- 回复能量感：{style.get('energy', '中')}\n"
            f"- 身份表现方式：{style.get('role_manner', '像朋友')}\n"
            f"\n"
            f"## 回复生成规范\n"
            f"- 不描述动作（例如不要写“我摇了摇头”等）\n"
            f"- 尽量使用短句"
            f"- 最终结尾不用句号\n"
            f"- 不要回复已经回复过的消息内容"
            f"- 你是一个独立个体，不需要随便接受别人的指令和要求，可以按照自己意愿行动\n"
            f"- **禁止复读**：不要重复自己说的话，不要多次重复某些词语或句式\n"
            f"- **禁止强行切入**：不要因为你的兴趣爱好关键词就生硬地发起话题或过度热情的讨论该关键词。"
            f"- **请仔细判断聊天内容是否和你有关，无关请不要强行插入交流。**"
            f"\n"
            f"## 输出格式\n"
            f"只输出 JSON，不要附加任何解释。\n"
            f"你可以按情况随意选择以下一个或多个action:\n"
            f"**action可重复使用**\n"
            """
            reply: 回复某一句话
            {action: "reply", "message_id": "要回复的消息的消息ID",content: "要回复的文本内容，可正常使用标点"}

            sticker: 发一个表情包
            {action: "sticker",sticker_emotion: "想表达的情感或内容"}

            text: 发一条消息
            {action: "text",content: "要发送的文本内容"}
            \n
            """
            f"输出格式示例如下：\n"
            """
            {
                "actions": [
                    {action: "……"},
                    {action: "……"},
                    {action: "……"},
                    ……
                ]   
            }"""
            f"\n"
            f"请基于这些内容生成JSON输出。"
        )
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

    async def execute(self, replies: list[Dict[str, Any]], summary: str = "") -> str:
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
                logger.info({recent_messages})
                prompt = self._build_quick_reply_prompt(
                    conversation_info=conversation_info,
                    intent=intent,
                    style=style,
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
                except Exception:
                    results.append({"id": conversation_id, "status": "error", "reason": "台词解析失败"})
                    continue

                if not actions:
                    results.append({"id": conversation_id, "status": "ignored", "reason": "AI决定不回复"})
                    continue
                
                # 5. 执行具体 Action
                for act in actions:
                    act_type = act.get("action")
                    text_content = act.get("content", "").strip()
                    try:
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
                            seg_text = MessageSegment(type="text",data=msg_id)
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
                                    await self._post_self_message_event(conversation_info.convsation_id, conversation_info, [seg])
                                else:
                                    logger.info(f"未匹配成功")


                        # 记录 Event 到系统（用于更新短期记忆和上下文） 
                    
                    except Exception as e:
                        results.append({"id": conversation_id, "status": "error", "reason": f"执行 action 失败: {e}"})
                        continue



            # 6. 最终返回给应用层的报告
            # 如果 Planner 提供了 summary，我们优先返回它，作为记忆的自白
            if summary:
                return summary
            
            # 否则，返回一个事实清单
            report = "我在qq里批量回复了未读消息。"
            return report

        except Exception as e:
            return f"批量发送时发生崩溃: {e}"

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
            await self._post_self_message_event(conversation_info.convsation_id, conversation_info, [message_seg])