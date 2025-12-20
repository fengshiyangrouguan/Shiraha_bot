import time
from typing import Dict, Any, List, Optional

from sqlmodel import select

from src.cortices.tools_base import BaseTool
from src.agent.world_model import WorldModel
from src.cortices.qq_chat.chat.qq_chat_data import QQChatData
from src.cortices.qq_chat.chat.chat_stream import QQChatStream
from src.common.database.database_manager import DatabaseManager
from src.system.di.container import container
from src.common.database.database_model import EventDB

class ViewUnreadMsgTool(BaseTool):
    """
    查看指定聊天中所有未读消息。
    该工具会从数据库中获取从上次阅读时间到现在的消息，并会将这些消息标记为已读。
    """
    def __init__(self, world_model: WorldModel):
        self._world_model = world_model

    @property
    def scope(self) -> str:
        return "main"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "view_unread_message",
            "description": "查看所有未读消息，并会将这些消息标记为已读。",
            "parameters": {},
            "required_parameters": []
        }

    def format_unread_messages(self,messages: List[EventDB]) -> List[str]:
        formatted_messages = []
        for msg in messages:
            sender_cardname = msg.user_cardname or "未知用户"
            sender_nickname = msg.user_nickname or "未知用户"
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.time))
            content = msg.event_content or "[消息内容为空]"
            line = f"[{timestamp_str}] {sender_cardname}({sender_nickname}): {content}"
            formatted_messages.append(line)
        return formatted_messages

    async def execute(self) -> str:
        """
        获取全部未读消息
        """
        try:
            qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
            if not qq_chat_data:
                return "qq未连接，没法看聊天记录。"
            
            # 获取所有有未读消息的聊天流
            unread_stream = []   # (stream, chat_stream)
            for stream_id, chat_stream in qq_chat_data.streams.items():
                if chat_stream.unread_count > 0:
                    unread_stream.append((stream_id, chat_stream))

            if not unread_stream:
                return "当前没有任何未读消息。请不要重复调用该工具"


            # 先收集全部未读消息
            all_formatted_lines = []
            database_manager = container.resolve(DatabaseManager)
            chat_stream: QQChatStream
            for stream_id, chat_stream in unread_stream:

                query = (
                    select(EventDB)
                    .where(EventDB.conversation_id == chat_stream.conversation_info.conversation_id)
                    .order_by(EventDB.time.desc())
                    .limit(chat_stream.unread_count)
                )

                
                messages_desc: List[EventDB] = await database_manager.get_all(query)
                messages = list(reversed(messages_desc)) # 反转列表，以实现时间正序
                formatted_messages = self.format_unread_messages(messages)
                stream_name = chat_stream.conversation_info.conversation_name
                conversation_type = chat_stream.conversation_info.conversation_type
                type = "群聊"
                if conversation_type == "private":
                    type = "私聊"
                    
                all_formatted_lines.append(f"====== {type} {stream_name} 未读 {len(messages)} 条 ======")
                all_formatted_lines.extend(formatted_messages)
                all_formatted_lines.append("\n")

                # 标记为已读
                chat_stream.mark_as_read()

            await self._world_model.save_cortex_data("qq_chat_data", qq_chat_data)
            return "\n".join(all_formatted_lines)


        except Exception as e:
            return f"获取未读消息时出错: {e}"
