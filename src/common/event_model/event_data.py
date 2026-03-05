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
                texts.append(f"[回复{seg.data}的消息]:")

            elif seg.type == "at":
                #适配器传回的就是 @ xxx 的文本
                texts.append(f"@{seg.data} ")

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
                    texts.append(f"发了一张图片：[{description}]")
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
                    prompt = "这是一个表情包，请详细描述一下表情包所表达的情感和内容，简短描述细节，从互联网梗,meme的角度去分析，精简回答"
                    # 调用API获取图片描述
                    description, model_name = await vlm_request.execute_with_image(
                        prompt=prompt,
                        base64_image_data=seg.data
                    )
                    texts.append(f"发了一个表情包：[{description}]")
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



        from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional, Dict, Literal
from abc import ABC, abstractmethod
import hashlib
import base64
import os

from src.llm_api.factory import LLMRequestFactory
from src.system.di.container import container
# 延迟导入，仅在需要时（即处理新表情包时）导入，避免循环依赖问题
# from src.cortices.qq_chat.chat.sticker_system.sticker_manager import StickerManager
# from src.common.database.database_manager import DatabaseManager
# from src.common.database.database_model import StickerDB
# from src.common.event_model.sticker import Sticker

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
        llm_factory: LLMRequestFactory = container.resolve(LLMRequestFactory)
        
        # 尝试从容器解析 StickerManager，如果失败则说明当前上下文不涉及表情包
        sticker_manager = None
        try:
            from src.cortices.qq_chat.chat.sticker_system.sticker_manager import StickerManager
            sticker_manager = container.resolve(StickerManager)
        except Exception:
            pass # Not in QQ cortex context, fine.


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
                texts.append(f"[回复{seg.data}的消息]:")

            elif seg.type == "at":
                #适配器传回的就是 @ xxx 的文本
                texts.append(f"@{seg.data} ")

            elif seg.type == "image":
                try:
                    image_base64 = seg.data
                    image_bytes = base64.b64decode(image_base64)
                    file_hash = hashlib.sha256(image_bytes).hexdigest()
                    
                    existing_sticker = None
                    if sticker_manager:
                        existing_sticker = sticker_manager.get_sticker_by_hash(file_hash)

                    if existing_sticker:
                        # --- 找到了已知表情包 ---
                        seg.type = "sticker" # 将类型转换为 sticker
                        keywords_text = ",".join(existing_sticker.keywords)
                        texts.append(f"发了一个表情包：[{keywords_text}]")
                    else:
                        # --- 未找到，作为新图片处理（并可能创建为新表情包） ---
                        vlm_request = llm_factory.get_request("vlm")
                        prompt = "这是一个表情包，请详细描述一下表情包所表达的情感和内容，简短描述细节，从互联网梗,meme的角度去分析，精简回答"
                        description, _ = await vlm_request.execute_with_image(
                            prompt=prompt,
                            base64_image_data=image_base64
                        )
                        texts.append(f"发了一个表情包：[{description}]")
                        
                        # 如果在 sticker_manager 上下文中，则创建新表情包
                        if sticker_manager:
                            from src.common.database.database_manager import DatabaseManager
                            from src.common.database.database_model import StickerDB
                            from src.common.event_model.sticker import Sticker

                            db_manager = container.resolve(DatabaseManager)
                            sticker_dir = "data/qq_chat_stickers" # 保持与cortex中一致

                            # 创建 Sticker 对象
                            new_sticker = Sticker(
                                file_hash=file_hash,
                                file_format="png", # 简化处理
                                keywords=description.split("，") # 简单用逗号分割描述作为关键词
                            )
                            file_path = os.path.join(sticker_dir, f"{new_sticker.sticker_id}.png")

                            # 保存文件
                            with open(file_path, "wb") as f:
                                f.write(image_bytes)
                            
                            # 保存到数据库
                            sticker_db_record = StickerDB(
                                id=new_sticker.sticker_id,
                                image_hash=new_sticker.file_hash,
                                full_path=file_path,
                                format=new_sticker.file_format,
                                description=description,
                                emotion=",".join(new_sticker.keywords),
                                record_time=str(new_sticker.created_at),
                                usage_count='0'
                            )
                            async with db_manager.get_session() as session:
                                session.add(sticker_db_record)
                                await session.commit()
                            
                            # 更新内存缓存
                            sticker_manager.add_sticker_to_cache(new_sticker)
                            seg.type = "sticker"

                except Exception as e:
                    texts.append(f"[{seg.type}]")
                    print(f"为图片生成描述或处理为表情包时失败: {e}")

            elif seg.type == "sticker":
                # 如果类型已经是 sticker (可能来自旧数据或已被处理)
                texts.append(f"发了一个表情包：[{seg.data}]")

            else:
                # 其它类型，用占位表示
                texts.append(f"[{seg.type}]")

        # 拼接结果并缓存
        self.LLM_plain_text = "".join(texts)



    def __repr__(self):
        return f"<message_id: {self.message_id} 消息内容: '{self.segments}'>"