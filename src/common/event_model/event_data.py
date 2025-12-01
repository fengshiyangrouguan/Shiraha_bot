from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional, Dict, Literal
from abc import ABC, abstractmethod

from src.llm_api.factory import LLMRequestFactory
from src.system.di.container import container


@dataclass
class BaseEventData(ABC):
    """
    所有eventdata类的抽象基类。
    定义了消息对象必须具备的核心属性和方法。
    """
    
    LLM_plain_text: Optional[str] = field(default=None, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    """解析完拼接的纯文本，用于短期记忆生成"""
    
    @abstractmethod
    async def process_to_context(self):
        """
        将消息段解析为 LLM 可用的纯文本上下文 (LLM_plain_text)。
        此方法需要将 @提及、图片等非文本段转换为 LLM 友好的文本描述。
        """
        pass
    

@dataclass
class MessageSegment:
    """
    标准化消息段数据类。
    消息内容可以由多个消息段组成，例如文本、图片、@信息等。
    """
    type: str
    """消息段类型 (e.g., "text", "image", "at", "poke")"""
    data: any
    """消息段的具体数据"""



@dataclass
class Message(BaseEventData):
    """
    所有平台的标准化消息模型类。
    """
    
    # 消息 ID（平台提供或适配器生成）
    message_id: Optional[str] = None
    # 消息内容片段列表（text/image/mention 等）
    segments: List[MessageSegment] = field(default_factory=list)


    def add_segment(self, segment: MessageSegment):
        """向消息中追加一个消息段。"""
        self.segments.append(segment)
        self.LLM_plain_text = None  # invalidate cache


    async def process_to_context(self):
        """
        将消息段解析为纯文本，并存储到 LLM_plain_text。
        文本段直接拼接，@ 提及、emoji、语音、图片 等类型转为文本，再拼接进来。
        """
        texts = []
        llm_factory = container.resolve(LLMRequestFactory)

        for seg in self.segments:
            if seg.type == "text":
                texts.append(str(seg.data))
            elif seg.type == "face":
                # emoji 直接当作文本拼接
                texts.append(str(seg.data))

            elif seg.type == "forward":
                # TODO: 这里未来可以先从adapter 加一个转发消息递归解析，然后调用LLM生成描述
                texts.append(f"[转发消息概述:{seg.data}]")

            elif seg.type == "reply":
                texts.append(f"[回复{seg.data}的消息{self.metadata.get("reply_text", "")}]:")

            elif seg.type == "at":
                #适配器传回的就是 @ xxx 的文本
                texts.append(f"{seg.data}: ")

            elif seg.type == "image":
                try:
                    # 获取VLM请求实例
                    vlm_request = llm_factory.get_request("vlm")
                    # 定义一个标准的、非对话式的提示词
                    prompt = "请仅简单描述图片的视觉内容，不要添加任何额外的问候、感想或建议。"
                    # 调用API获取图片描述
                    description, model_name = await vlm_request.execute_with_image(
                        prompt=prompt,
                        base64_image_data=seg.data
                    )
                    texts.append(f"[图片描述: {description}]")
                except Exception as e:
                    # 如果生成描述失败，使用旧的占位符
                    texts.append(f"[{seg.type}]")
                    # 可以考虑记录错误日志
                    print(f"为图片生成描述失败: {e}")
            elif seg.type == "sticker":
                try:
                    # 获取VLM请求实例
                    vlm_request = llm_factory.get_request("vlm")
                    # 定义一个标准的、非对话式的提示词
                    prompt = "请仅简单描述这个表情包的视觉内容，不要添加任何额外的问候、感想或建议。"
                    # 调用API获取图片描述
                    description, model_name = await vlm_request.execute_with_image(
                        prompt=prompt,
                        base64_image_data=seg.data
                    )
                    texts.append(f"发了一个表情包：[表情包描述: {description}]")
                except Exception as e:
                    # 如果生成描述失败，使用旧的占位符
                    texts.append(f"[{seg.type}]")
                    # 可以考虑记录错误日志
                    print(f"为表情包生成描述失败: {e}")
            else:
                # 其它类型，用占位表示
                texts.append(f"[{seg.type}]")

        # 拼接结果并缓存
        self.LLM_plain_text = "".join(texts)



    def __repr__(self):
        return f"<message_id: {self.message_id} 消息内容: '{self.segments}'>"