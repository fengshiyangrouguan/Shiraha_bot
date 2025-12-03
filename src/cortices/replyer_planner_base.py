# src/cortices/replyer_planner_base.py
from abc import ABC, abstractmethod
from typing import List, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.platform.platform_base import BasePlatformAdapter, MessageSegment

class ReplyIntent:
    """
    表示来自上层 Planner 的回复意图。
    这是一个数据类，用于封装回复所需的所有信息。
    """
    def __init__(self, text: str = "", at_users: List[str] = None, images: List[str] = None):
        self.text = text
        self.at_users = at_users or []
        self.images = images or []
        # 未来可以扩展，例如 attachments, voice 等