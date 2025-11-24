import logging
import time
from typing import Any, Dict, List, TYPE_CHECKING

from src.platform.sources.QQ_napcat.utils import forward, image
from src.common.event_model.info_data import UserInfo, ConversationInfo
from src.common.event_model.event import Event
from src.common.event_model.event_data import (
    Message, 
    MessageSegment
)

# 仅用于类型提示，防止循环导入
if TYPE_CHECKING:
    from src.platform.sources.QQ_napcat.adapter import QQNapcatAdapter



logger = logging.getLogger(__name__)


class NapcatEventDispatcher:

    def __init__(self, adapter:"QQNapcatAdapter"):
        self.adapter = adapter

    async def dispatch(self, raw_event: Dict[str, Any]):
        """
        分发 Napcat 原始事件
        """

        post_type = raw_event.get("post_type")

        # -------------------------------
        # 1. 消息事件
        # -------------------------------
        if post_type == "message":
            event = await self._handle_message_event(raw_event)
            await self.adapter.post_method(event)
            return

        # -------------------------------
        # 2. 通知事件 (group increase, file upload …)
        # -------------------------------
        if post_type == "notice":
            # 先pass
            return
            event = self._handle_notice_event(raw_event)
            await self.adapter.post_method(event)
            return


        # 未知事件不管
        # logger.debug(f"无法处理的事件类型：{post_type}")

    

    async def _handle_message_event(self,raw_event: Dict[str, Any]) -> Message:
        """
        将 OneBot 原始消息解析为 Message 对象
        """
        # 构建消息段
        user_info:UserInfo = self._parse_user_info(raw_event)
        conversation_info:ConversationInfo = self._parse_conversation_info(raw_event)
        segments: List[MessageSegment] = []
        for seg in raw_event.get("message", []):

            seg_type = seg.get("type")
            data = seg.get("data", {})
            
            if seg_type == "text":
                data = data["text"]

            elif seg_type == "image":
                # 判断是不是表情包

                summary = seg["data"].get("summary", "")
                if summary == "[动画表情]": 
                    seg_type = "sticker"
                file_id = data["file"]
                file_size = data["file_size"]
                #TODO: 应在获取消息前加一个判断是否尺寸过大，加一个尺寸解析阈值
                
                base_64 = await image.get_image_base64_async(data["url"])
                # data = {"file_id":file_id,"file_size":file_size,"base64":base_64}
                data = base_64

            elif seg_type == "face":
                raw_data = seg.get("data", {})
                data = raw_data.get("id", {})

            elif seg_type == "forward":
                data = forward.extract_forward_info_from_raw(raw_event)

            #TODO 当前无特殊解析逻辑的直接插入 例如reply，at，json，以后再做处理
            segments.append(MessageSegment(type=seg_type, data=data))

        # 构建 Message
        message = Message(
            message_id=str(raw_event.get("message_id")),
            segments=segments,
            raw_message=raw_event
        )
        
        event = Event(
            event_type="message",
            event_id=str(object=raw_event.get("message_id", time.time())),
            time=raw_event.get("time", int(time.time())),
            platform="qq",
            conversation_info=conversation_info,
            user_info=user_info,
            event_data=message
        )

        return event

    def _parse_user_info(self,raw_event: Dict[str, Any]) -> UserInfo:
        sender:Dict = raw_event.get("sender", {})
        user_info = UserInfo(
                        user_id=str(raw_event["user_id"]),        # 统一转为 str
                        user_nickname=sender.get("nickname") or None,    # 空字符串转为 None
                        user_cardname=sender.get("card") or None         # 空字符串转为 None
                    )
        return user_info

    def _parse_conversation_info(self,raw_event: Dict[str, Any]) -> ConversationInfo:
        """从 OneBot raw_event 中解析 ConversationInfo。"""
        message_type = raw_event.get("message_type")
        platform_meta = {}
        if message_type == "private":
            sender:Dict = raw_event.get("sender", {})
            conversation_id = str(raw_event["user_id"])
            name = sender.get("nickname") or None,    # 空字符串转为 None
            parent_id = None                          # OneBot 群聊无父级概念

        elif message_type == "group":
            conversation_id = str(raw_event["group_id"])
            raw_data: Dict = raw_event.get("raw", {})
            name = raw_data.get("peerName") or None
            parent_id = None  # OneBot 群聊无父级概念

        conversation_info = ConversationInfo(
                                conversation_id=conversation_id,
                                conversation_type=message_type,  # type: ignore  # 因已校验，安全
                                conversation_name=name,
                                parent_id=parent_id,
                                platform_meta=platform_meta

                            )
        return conversation_info
        
    def _handle_notice_event(self,raw_event: Dict[str, Any]):
        pass

