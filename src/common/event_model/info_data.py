# src/core/entities.py
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict, Literal

@dataclass
class UserInfo:
    """
    标准化用户信息数据类。
    用于封装来自不同平台的发送者信息，提供统一的访问接口。
    """
    user_id: str   # 用户在该平台上的唯一ID
    user_nickname: Optional[str] = None  # 用户的昵称
    user_cardname: Optional[str] = None  # 用户在群组中的名片/群昵称 (可选)

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class ConversationInfo:
    """
    标准化“对话空间”信息数据类。
    用于封装来自不同平台的会话信息，提供统一的访问接口。

    它是跨平台的统一抽象，适用于：
      - 私聊（private）
      - 群聊（group）
      - 频道（channel）
      - 子线程（thread）

    字段说明：
        conversation_id: 当前对话空间的唯一 ID（频道ID/群ID/私聊ID/thread ID）
        conversation_type: 对话类型，用于决定处理方式
        name: 可读名称（频道名/群名/线程名）
        parent_id: 父级对话 ID（线程 → 频道；TG Forum topic → group）
        platform_meta: 平台特定的附加信息，不做跨平台语义解析
    """

    conversation_id: str
    conversation_type: Literal["private", "group", "channel", "thread"]

    # 便于展示的名字，不参与逻辑
    conversation_name: Optional[str] = None

    # 对于 thread / forum topic 等结构化对话十分关键
    parent_id: Optional[str] = None

    # 平台特有字段（例如 Discord 有 guild_id）
    platform_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转为可 JSON 持久化的结构"""
        return asdict(self)


