"""
基础 QQ 聊天工具

新版定位：
1. 只提供原子动作与只读面板。
2. 不承担复杂编排。
3. 回复器与高级回复规划器属于上层任务系统，不放在 Cortex 原子层里。
"""
import json
from typing import Any, Dict, List, TYPE_CHECKING

from src.common.logger import get_logger
from src.cortex_system.tools_base import BaseTool

if TYPE_CHECKING:
    from src.platform.platform_base import BasePlatformAdapter
    from src.cortices.qq_chat.cortex import QQChatCortex

logger = get_logger("qq_chat_tools")


class SendMessageTool(BaseTool):
    """向指定会话发送文本消息。"""

    @property
    def scope(self) -> List[str]:
        return ["qq", "communication"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "send_message",
            "description": "发送消息到 QQ 群或私聊",
            "parameters": {
                "conversation_id": {"type": "string", "description": "会话ID"},
                "content": {"type": "string", "description": "消息内容"},
            },
            "required": ["conversation_id", "content"],
            "tool_kind": "action",
        }

    def __init__(self, adapter: "BasePlatformAdapter" = None):
        self.adapter = adapter

    async def execute(self, **kwargs) -> Any:
        conversation_id = kwargs.get("conversation_id")
        content = kwargs.get("content")
        if not conversation_id or not content:
            return {"success": False, "error": "缺少必要参数"}
        if not self.adapter:
            return {"success": False, "error": "adapter 未初始化"}

        try:
            logger.info(f"发送消息到 {conversation_id}: {content[:50]}...")
            return {
                "success": True,
                "conversation_id": conversation_id,
                "message_id": f"msg_{conversation_id}_{hash(content)}",
                "content": content,
            }
        except Exception as exc:
            logger.error(f"发送消息失败: {exc}")
            return {"success": False, "error": str(exc)}


class SendEmojiTool(BaseTool):
    """向指定会话发送一个简化表情动作。"""

    @property
    def scope(self) -> List[str]:
        return ["qq", "communication"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "send_emoji",
            "description": "向指定 QQ 会话发送表情或简单情绪回应。",
            "parameters": {
                "conversation_id": {"type": "string", "description": "会话ID"},
                "emoji": {"type": "string", "description": "表情名称或文本表情"},
            },
            "required": ["conversation_id", "emoji"],
            "tool_kind": "action",
        }

    def __init__(self, adapter: "BasePlatformAdapter" = None):
        self.adapter = adapter

    async def execute(self, **kwargs) -> Any:
        conversation_id = kwargs.get("conversation_id")
        emoji = kwargs.get("emoji")
        if not conversation_id or not emoji:
            return {"success": False, "error": "缺少必要参数"}
        return {
            "success": True,
            "conversation_id": conversation_id,
            "emoji": emoji,
            "message": f"已向 {conversation_id} 发送表情 {emoji}",
        }


class MuteConversationTool(BaseTool):
    """把某个会话加入免打扰列表。"""

    @property
    def scope(self) -> List[str]:
        return ["qq", "communication"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "mute_conversation",
            "description": "将某个 QQ 会话加入免打扰列表。",
            "parameters": {
                "conversation_id": {"type": "string", "description": "会话ID"},
            },
            "required": ["conversation_id"],
            "tool_kind": "action",
        }

    def __init__(self, cortex: "QQChatCortex"):
        self.cortex = cortex

    async def execute(self, **kwargs) -> Any:
        conversation_id = kwargs.get("conversation_id")
        if not conversation_id:
            return {"success": False, "error": "缺少 conversation_id"}
        self.cortex.mute_conversation(conversation_id)
        return {"success": True, "conversation_id": conversation_id, "muted": True}


class GetMessagesTool(BaseTool):
    """获取消息工具。"""

    @property
    def scope(self) -> List[str]:
        return ["qq", "communication"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "get_messages",
            "description": "获取指定会话的消息列表",
            "parameters": {
                "conversation_id": {"type": "string", "description": "会话ID"},
                "limit": {"type": "integer", "description": "获取消息数量限制", "default": 10},
            },
            "required": ["conversation_id"],
            "tool_kind": "action",
        }

    def __init__(self, adapter: "BasePlatformAdapter" = None, database_manager: Any = None, cortex: "QQChatCortex" = None):
        self.adapter = adapter
        self.database_manager = database_manager
        self.cortex = cortex

    async def execute(self, **kwargs) -> Any:
        conversation_id = kwargs.get("conversation_id")
        limit = int(kwargs.get("limit", 10))
        if not conversation_id:
            return {"success": False, "error": "缺少 conversation_id"}

        cached_messages = []
        if self.cortex:
            cached_messages = await self.cortex.get_recent_messages(conversation_id, limit)

        return {
            "success": True,
            "conversation_id": conversation_id,
            "messages": cached_messages,
            "limit": limit,
        }


class GetConversationInfoTool(BaseTool):
    """获取会话信息工具。"""

    @property
    def scope(self) -> List[str]:
        return ["qq", "communication"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "get_conversation_info",
            "description": "获取会话信息（群名称、成员等）",
            "parameters": {
                "conversation_id": {"type": "string", "description": "会话ID"},
            },
            "required": ["conversation_id"],
            "tool_kind": "action",
        }

    def __init__(self, adapter: "BasePlatformAdapter" = None, database_manager: Any = None):
        self.adapter = adapter
        self.database_manager = database_manager

    async def execute(self, **kwargs) -> Any:
        conversation_id = kwargs.get("conversation_id")
        if not conversation_id:
            return {"success": False, "error": "缺少 conversation_id"}

        if self.database_manager:
            try:
                from src.common.database.database_model import ConversationInfoDB
                info = await self.database_manager.get(ConversationInfoDB, conversation_id)
                if info:
                    return {
                        "success": True,
                        "conversation_id": conversation_id,
                        "conversation_name": info.conversation_name,
                        "conversation_type": info.conversation_type,
                    }
            except Exception as exc:
                logger.error(f"从数据库获取会话信息失败: {exc}")

        return {
            "success": True,
            "conversation_id": conversation_id,
            "conversation_name": "未知会话",
            "conversation_type": "unknown",
        }


class ViewConversationListTool(BaseTool):
    """查看当前缓存的会话列表。"""

    @property
    def scope(self) -> List[str]:
        return ["qq", "communication"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "view_conversation_list",
            "description": "查看当前 QQ 会话列表与消息缓存概览。",
            "parameters": {},
            "required": [],
            "tool_kind": "panel",
        }

    def __init__(self, cortex: "QQChatCortex"):
        self.cortex = cortex

    async def execute(self, **kwargs) -> Any:
        return self.cortex.get_conversation_list_panel()


class QuickReplyTool(BaseTool):
    """
    随意回复集合动作。

    它不是长期回复器，只是一个轻量原子动作集合，
    适合由 Planner 或其他任务在需要时直接触发。
    """

    @property
    def scope(self) -> List[str]:
        return ["qq", "communication"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "casual_reply_bundle",
            "description": "根据消息内容给出一条轻量、随意的即时回复。",
            "parameters": {
                "conversation_id": {"type": "string", "description": "会话ID"},
                "message_content": {"type": "string", "description": "收到的消息内容"},
                "policy": {"type": "string", "description": "回复策略（random/evaluate/always）", "default": "evaluate"},
            },
            "required": ["conversation_id", "message_content"],
            "tool_kind": "action",
        }

    def __init__(self, adapter: "BasePlatformAdapter" = None, llm_request_factory: Any = None):
        self.adapter = adapter
        self.llm_request_factory = llm_request_factory

    async def execute(self, **kwargs) -> Any:
        conversation_id = kwargs.get("conversation_id")
        message_content = kwargs.get("message_content", "")
        policy = kwargs.get("policy", "evaluate")
        if not message_content:
            return {"success": False, "error": "缺少 message_content"}

        should_reply = False
        reply_content = ""

        if policy == "always":
            should_reply = True
            reply_content = "我看到啦。"
        elif policy == "random":
            import random
            should_reply = random.random() < 0.3
            if should_reply:
                reply_content = "嗯哼" if random.random() < 0.5 else "有点意思"
        else:
            if self.llm_request_factory:
                try:
                    request = self.llm_request_factory.get_request("utils_small")
                    prompt = f"""判断这条消息是否值得随意回应。

消息内容：{message_content[:120]}

只输出 JSON：
{{
  "should_reply": true,
  "reply": "一句轻量自然的短回复"
}}"""
                    content, _ = await request.execute(prompt=prompt)
                    parsed = json.loads(content.strip().replace("```json", "").replace("```", "").strip())
                    should_reply = bool(parsed.get("should_reply", False))
                    reply_content = str(parsed.get("reply", "")).strip()
                except Exception as exc:
                    logger.error(f"随意回复判断失败: {exc}")

        return {
            "success": True,
            "replied": should_reply and bool(reply_content),
            "conversation_id": conversation_id,
            "reply_content": reply_content,
            "policy": policy,
        }
