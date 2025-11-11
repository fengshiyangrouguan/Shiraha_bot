# core/chat.py
import time
from typing import List, Dict, Any, Optional

class Message:
    """
    一个标准化的消息对象，用于统一处理不同来源的消息。
    """
    def __init__(self, user_id: str, content: str, timestamp: Optional[float] = None):
        self.user_id = user_id
        self.content = content
        self.timestamp = timestamp or time.time()

    def __repr__(self):
        return f"Message(user_id='{self.user_id}', content='{self.content}')"

class ChatStream:
    """
    一个聊天流，代表一个独立的对话上下文（如一个私聊或一个群聊）。
    它包含了一个对话的所有消息记录，是实现短期记忆的基础。
    """
    def __init__(self, stream_id: str):
        self.stream_id = stream_id  # 唯一的ID，可以是 "private_{user_id}" 或 "group_{group_id}"
        self.messages: List[Message] = []
        self.max_history_size = 20  # 为了简化，我们先只保存最近20条消息

    def add_message(self, user_id: str, content: str):
        """向聊天流中添加一条新消息。"""
        message = Message(user_id=user_id, content=content)
        self.messages.append(message)
        
        # 维持消息历史记录的大小
        if len(self.messages) > self.max_history_size:
            self.messages.pop(0)

    def get_history_prompt(self) -> str:
        """将聊天记录格式化为可以放入Prompt的字符串。"""
        if not self.messages:
            return "这是对话的开始。"
        
        prompt_lines = []
        for msg in self.messages:
            role = "用户" if msg.user_id != "BOT" else "你"
            prompt_lines.append(f"{role}: {msg.content}")
            
        return "\n".join(prompt_lines)

    def __repr__(self):
        return f"ChatStream(stream_id='{self.stream_id}', messages_count={len(self.messages)})"

class ChatManager:
    """
    管理所有的ChatStream实例。
    """
    def __init__(self):
        self._streams: Dict[str, ChatStream] = {}

    def get_or_create_stream(self, stream_id: str) -> ChatStream:
        """
        根据stream_id获取一个已有的ChatStream，如果不存在则创建一个新的。
        """
        if stream_id not in self._streams:
            self._streams[stream_id] = ChatStream(stream_id=stream_id)
        return self._streams[stream_id]

# 创建一个全局的ChatManager实例，方便在项目各处调用
chat_manager = ChatManager()
