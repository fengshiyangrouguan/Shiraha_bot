from typing import List, Optional, Dict, Any

class ChatContext:
    """
    标准化后的消息模型，由 event_processor 解析后放入 ChatStream。
    """
    def __init__(self, message_id: str, user_info, conv_info, text: str, segments=None):
        self.message_id = message_id
        self.user_info = user_info
        self.conversation_info = conv_info
        self.text = text
        self.segments = segments or []

    def process_segments(self):
        # 默认实现：把所有 segment 拼接成文本
        parts = []
        for s in self.segments:
            if s["type"] == "text":
                parts.append(s["data"].get("text", ""))
        self.text = "".join(parts)


class ChatStream:
    """
    存储当前上下文消息，包括：
    - 历史解析后的 ChatMessage
    - 用户消息
    - 机器人生成的回复
    """

    def __init__(self):
        self.contexts: List[ChatContext] = []

    def add_context(self, msg: ChatContext):
        self.contexts.append(msg)

    def get_context_window(self, limit: int = 20) -> List[ChatContext]:
        return self.contexts[-limit:]

