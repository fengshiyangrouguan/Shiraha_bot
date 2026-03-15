from typing import Dict, Any
from src.platform.sources.qq_napcat.utils import image

def build_text_seg(text: str) -> Dict[str, Any]:
    """
    构造纯文本消息段。
    
    :param text: 文本内容，包括 Unicode 表情。
    """
    return {"type": "text", "data": {"text": text}}

def build_face_seg(id: int) -> Dict[str, Any]:
    """
    构造 QQ 原生小黄脸表情消息段。
    
    :param id: 表情 ID (0-170+)。
    """
    return {"type": "face", "data": {"id": id}}

def build_at_seg(qq_id: int | str) -> Dict[str, Any]:
    """
    构造 @ 指定用户的消息段。
    
    :param qq_id: 目标用户的 QQ 号。
    """
    return {"type": "at", "data": {"qq": str(qq_id)}}

def build_reply_seg(message_id: int | str) -> Dict[str, Any]:
    """
    构造回复指定消息的消息段。
    
    :param message_id: 要回复的消息 ID。
    """
    return {"type": "reply", "data": {"id": str(message_id)}}

def build_image_seg(encoded_image: str) -> Dict[str, Any]:
    """
    构造图片消息段。
    
    :param file: 文件名、绝对路径、或网络 URL。
    """
    # 推荐使用 'file' 或 'url' 字段，这里使用 'file' 兼容路径和 URL
    return {"type": "image", "data": {"file": f"base64://{encoded_image}"}} 

def build_sticker_seg(encoded_image: str) -> Dict[str, Any]:
    """
    构造图片消息段。
    
    :param file: 文件名、绝对路径、或网络 URL。
    """
    # 推荐使用 'file' 或 'url' 字段，这里使用 'file' 兼容路径和 URL
    data = encoded_image
    image_format = image.get_image_format(encoded_image)
    if image_format != "gif":
        data = image.convert_image_to_gif(encoded_image)
    return {
        "type": "image",
        "data": {
            "file": f"base64://{data}",
            "subtype": 1,
            "summary": "[动画表情]",
        },
    }



def build_record_seg(file: str, cache: bool = True) -> Dict[str, Any]:
    """
    构造语音消息段。
    
    :param file: 文件名、绝对路径、或网络 URL。
    :param cache: 是否使用缓存。
    """
    return {"type": "record", "data": {"file": file, "cache": cache}}

def build_poke_seg(qq_id: int | str) -> Dict[str, Any]:
    """
    构造“戳一戳”消息段。
    
    :param qq_id: 被戳的用户的 QQ 号。
    """
    return {"type": "poke", "data": {"qq": str(qq_id)}}