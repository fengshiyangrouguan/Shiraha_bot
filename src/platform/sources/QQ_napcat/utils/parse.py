from typing import Any, Dict, List, Optional
from src.common.event_model.event_data import Message, MessageSegment
from src.common.event_model.info_data import UserInfo, ConversationInfo
from src.platform.sources.QQ_napcat.utils import forward


def _parse_message_data(raw_event: Dict[str, Any]) -> Message:
    """
    将 OneBot 原始消息解析为 Message 对象
    """
    # 构建消息段
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
                seg_type = "Sticker"
            data = {"file_id":data["file"], "url":data["url"], "file_size":data["file_size"]}


        elif seg_type == "face":
            raw_data = seg.get("data", {})
            data = raw_data.get("id", {})

        elif seg_type == "forward":
            data = forward.extract_forward_info_from_raw(raw_event)

        
        segments.append(MessageSegment(type=seg_type, data=data))

    # 构建 Message
    message = Message(
        message_id=str(raw_event.get("message_id")),
        segments=segments,
        raw_message=raw_event
    )

    return message

def _parse_user_info(raw_event: Dict[str, Any]) -> UserInfo:
    sender:Dict = raw_event.get("sender", {})
    user_info = UserInfo(
                    user_id=str(raw_event["user_id"]),        # 统一转为 str
                    user_nickname=sender.get("nickname") or None,    # 空字符串转为 None
                    user_cardname=sender.get("card") or None         # 空字符串转为 None
                )
    return user_info

def _parse_conversation_info(raw_event: Dict[str, Any]) -> ConversationInfo:
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
    
