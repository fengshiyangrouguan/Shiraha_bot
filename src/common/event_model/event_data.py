from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional, Dict, Literal


@dataclass
class MessageSegment:
    """
    标准化消息段数据类。
    消息内容可以由多个消息段组成，例如文本、图片、@信息等。
    """
    type: str  # 消息段类型 (e.g., "text", "image", "at", "poke")
    data: any  # 消息段的具体数据



@dataclass
class Message:
    """
    所有平台的标准化消息模型类。
    """
    
    # 消息 ID（平台提供或适配器生成）
    message_id: str
    
    # 消息内容片段列表（text/image/mention 等）
    segments: List[MessageSegment] = field(default_factory=list)

    # 平台消息原始信息
    raw_message: Optional[Dict[str, Any]] = None

    # 解析完拼接的纯文本，用于短期记忆生成
    LLM_plain_text: Optional[str] = None

    # ---------------------------------------------------------------------
    # 基础功能：给 pipeline/插件用
    # ---------------------------------------------------------------------

    def add_segment(self, segment: MessageSegment):
        """向消息中追加一个消息段。"""
        self.segments.append(segment)
        self._cached_plain_text = None  # invalidate cache

    def get_plain_text(self) -> str:
        """
        从所有消息段中提取纯文本。
        文本段直接拼接，非文本段忽略。
        """
        if self.LLM_plain_text is not None:
            return self.LLM_plain_text
        
        texts = []
        for seg in self.segments:
            if seg.type == "text":
                texts.append(str(seg.data))
        
        self.LLM_plain_text = "".join(texts)
        return self.LLM_plain_text

    @property
    def plain_text(self) -> str:
        return self.get_plain_text()

    # ---------------------------------------------------------------------
    # 处理消息段方法
    # ---------------------------------------------------------------------
    
    async def process_segments(self):
        """
        将消息段解析为纯文本，并存储到 LLM_plain_text。
        文本段直接拼接，@ 提及、emoji、语音、图片 等类型转为文本，再拼接进来。
        """
        texts = []

        for seg in self.segments:
            if seg.type == "text":
                texts.append(str(seg.data))
            elif seg.type == "emoji":
                # emoji 直接当作文本拼接
                texts.append(str(seg.data))
            elif seg.type == "mention":
                # 假设 data 包含 username
                username = seg.data.get("username", "")
                texts.append(f"@{username}")
            else:
                # 其它类型，用占位表示
                texts.append(f"[{seg.type}]")

        # 拼接结果并缓存
        self.LLM_plain_text = "".join(texts)



    def __repr__(self):
        return f"<message_id: {self.message_id} 消息内容: '{self.segments}'>"