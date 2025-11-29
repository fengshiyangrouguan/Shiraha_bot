import asyncio
from typing import List, Optional, Dict, Any

from src.core.chat.chat_stream import ChatContext, ChatStream
from src.common.event_model.event import Event


class ChatManager:
    """
    管理多个会话的 chat stream，并给 event_processor 调用。

    event_processor 会把 event 解析成 ChatContext，然后调用：
        chat_manager.append_message(conv_id, chat_message)
    """

    def __init__(self):
        self._streams: Dict[str, ChatStream] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def get_chat_stream(self, stream_id: str) -> ChatStream:
        if stream_id not in self._streams:
            self._streams[stream_id] = ChatStream()
            self._locks[stream_id] = asyncio.Lock()
        return self._streams[stream_id]

    async def append_context(self, stream_id: str, event: Event):
        #TODO: 在这写一个event转为context的逻辑，应该可以在event里定义，然后event调用base_event_data的trans方法
        lock = self._locks.setdefault(stream_id, asyncio.Lock())
        async with lock:
            stream: ChatStream = self.get_chat_stream(stream_id)
            stream.add_chat_context(event)

    async def get_all_context(self, stream_id: str, limit: int = 12) -> List[ChatContext]:
        lock = self._locks.setdefault(stream_id, asyncio.Lock())
        async with lock:
            stream: ChatStream = self.get_chat_stream(stream_id)
            return stream.get_context_window(limit)
