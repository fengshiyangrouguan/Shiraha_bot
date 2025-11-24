import uuid
from typing import Any, Dict, List, Callable

from src.common.event_model.event import Event
from src.common.event_model.info_data import ConversationInfo
from src.platform.sources.QQ_napcat.utils.msg_api_build import (
    build_text_seg,
    build_reply_seg,
    build_at_seg,
    build_face_seg,
    build_image_seg,
    build_poke_seg,
    build_record_seg
)

_send_websocket: Callable[[Dict], None]

class NapcatMessageService:

    def __init__(self, adapter):
        self.adapter = adapter

    # ======== 顶层api ========
    # 可以直接调用的最简单api，想要发送组合消息需要自行组合segement
    async def send_text(self, 
                        conversation_info: ConversationInfo, 
                        text: str, 
                        face_id: int | None = None,
                        reply_id: int | None = None,
                        at_id: Any | None = None):
        """
        发送文本消息，并根据可选参数附加 QQ 原生表情、回复或 @ 发送者。
        该方法将高层业务请求转换为 OneBot V11 消息段 (segments)，并自动路由发送。

        Args:
            conversation_info (ConversationInfo): 
                目标会话信息，包含 conversation_type (group/private) 和 conversation_id。
            text (str): 
                要发送的纯文本内容。Unicode 表情可以直接包含在内。
            face_id (int | None): 
                可选。QQ 原生表情的 ID（如 123）。如果提供，将作为单独的消息段附加。默认为 None。
            reply_id (int | None): 
                可选。是否以“回复”模式发送消息。如果传入message_id则使用回复模式。
            at_id (Any \ None): 
                可选。是否在消息中 @ 触发当前 Event 的用户。如果传入id，将自动在文本前附加 @ 消息段，如果传入all,则@全体成员，默认为 None。

        Returns:
            Any | None: 
                发送 API 调用后的结果
        
        Raises:
            发送失败，抛出异常
        """

        segments = []

        # 1.判断是否回复模式
        if reply_id is not None:
            segments.append(build_reply_seg(reply_id))

        # 2.判断是否加at
        if at_id is not None:
            segments.append(build_at_seg(at_id))

        # 3. 添加文本段 (Text Segment)
        segments.append(build_text_seg(text))

        # 4. 添加表情段 (Face Segment)
        if face_id is not None:
            segments.append(build_face_seg(face_id))
        
        # 5. 调用底层api自动发送
        return await self.send_message(conversation_info, segments)


    async def send_image(self, conversation_info: ConversationInfo, file: str):
        pass
    async def send_sticker(self, conversation_info: ConversationInfo, file: str):
        pass



    # ======== 底层api ========
    async def send_message(self, conversation_info: ConversationInfo, segments: List[Dict[str, Any]]):
        """
        自动选择私聊 or 群聊
        event = MessageEvent
        """
        
        action: str = None
        target_id = conversation_info.conversation_id
        type = conversation_info.conversation_type
        id_key = {"group":"group_id","private":"user_id"}

        if type == "group":
            action: str = "send_group_msg"
        elif type == "private":
            action: str = "send_private_msg"

        payload = {
            "action": action,   
            "params": {
                id_key[type]: target_id,
                "message": segments
            },
        }
        await self.adapter._send_websocket(payload)
