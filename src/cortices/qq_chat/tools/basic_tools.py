"""
基础 QQ 聊天工具

提供最基础的 QQ 交互功能，无复杂逻辑链
"""
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from src.cortex_system.tools_base import BaseTool
from src.common.logger import get_logger

if TYPE_CHECKING:
    from src.platform.platform_base import BasePlatformAdapter

logger = get_logger("qq_chat_tools")


class SendMessageTool(BaseTool):
    """发送消息工具"""

    @property
    def scope(self) -> List[str]:
        return ["qq", "communication"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "send_message",
            "description": "发送消息到 QQ 群或私聊",
            "parameters": {
                "conversation_id": {
                    "type": "string",
                    "description": "会话ID（群号或私聊QQ号）"
                },
                "content": {
                    "type": "string",
                    "description": "消息内容"
                }
            },
            "required": ["conversation_id", "content"]
        }

    def __init__(self, adapter: "BasePlatformAdapter" = None):
        self.adapter = adapter

    async def execute(self, **kwargs) -> Any:
        conversation_id = kwargs.get("conversation_id")
        content = kwargs.get("content")

        if not conversation_id or not content:
            return {
                "success": False,
                "error": "缺少必要参数"
            }

        if not self.adapter:
            return {
                "success": False,
                "error": "adapter 未初始化"
            }

        try:
            # TODO: 实现实际的消息发送逻辑
            # await self.adapter.send_message(conversation_id, content)
            logger.info(f"发送消息到 {conversation_id}: {content[:50]}...")
            return {
                "success": True,
                "conversation_id": conversation_id,
                "message_id": f"msg_{conversation_id}_{hash(content)}"
            }
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class GetMessagesTool(BaseTool):
    """获取消息工具"""

    @property
    def scope(self) -> List[str]:
        return ["qq", "communication"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "get_messages",
            "description": "获取指定会话的消息列表",
            "parameters": {
                "conversation_id": {
                    "type": "string",
                    "description": "会话ID"
                },
                "limit": {
                    "type": "integer",
                    "description": "获取消息数量限制",
                    "default": 10
                }
            },
            "required": ["conversation_id"]
        }

    def __init__(self, adapter: "BasePlatformAdapter" = None, database_manager: Any = None):
        self.adapter = adapter
        self.database_manager = database_manager

    async def execute(self, **kwargs) -> Any:
        conversation_id = kwargs.get("conversation_id")
        limit = kwargs.get("limit", 10)

        if not conversation_id:
            return {
                "success": False,
                "error": "缺少 conversation_id"
            }

        # 尝试从数据库获取
        if self.database_manager:
            try:
                from src.common.database.database_model import EventDB
                # TODO: 实现数据库查询
                # events = await self.database_manager.get_all(...)
                logger.debug(f"从数据库获取 {conversation_id} 的消息")
            except Exception as e:
                logger.error(f"从数据库获取消息失败: {e}")

        return {
            "success": True,
            "conversation_id": conversation_id,
            "messages": [],  # TODO: 返回实际消息
            "limit": limit
        }


class GetConversationInfoTool(BaseTool):
    """获取会话信息工具"""

    @property
    def scope(self) -> List[str]:
        return ["qq", "communication"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "get_conversation_info",
            "description": "获取会话信息（群名称、成员等）",
            "parameters": {
                "conversation_id": {
                    "type": "string",
                    "description": "会话ID"
                }
            },
            "required": ["conversation_id"]
        }

    def __init__(self, adapter: "BasePlatformAdapter" = None, database_manager: Any = None):
        self.adapter = adapter
        self.database_manager = database_manager

    async def execute(self, **kwargs) -> Any:
        conversation_id = kwargs.get("conversation_id")

        if not conversation_id:
            return {
                "success": False,
                "error": "缺少 conversation_id"
            }

        # 尝试从数据库获取会话信息
        if self.database_manager:
            try:
                from src.common.database.database_model import ConversationInfoDB
                info = await self.database_manager.get(ConversationInfoDB, conversation_id)
                if info:
                    return {
                        "success": True,
                        "conversation_id": conversation_id,
                        "conversation_name": info.conversation_name,
                        "conversation_type": info.conversation_type
                    }
            except Exception as e:
                logger.error(f"从数据库获取会话信息失败: {e}")

        return {
            "success": True,
            "conversation_id": conversation_id,
            "conversation_name": "未知会话",
            "conversation_type": "unknown"
        }


class QuickReplyTool(BaseTool):
    """快速回复工具（使用小模型判断是否感兴趣）"""

    @property
    def scope(self) -> List[str]:
        return ["qq", "communication"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "quick_reply",
            "description": "快速回复消息（使用小模型判断是否回应）",
            "parameters": {
                "conversation_id": {
                    "type": "string",
                    "description": "会话ID"
                },
                "message_content": {
                    "type": "string",
                    "description": "收到的消息内容"
                },
                "policy": {
                    "type": "string",
                    "description": "回复策略（random/evaluate/always）",
                    "default": "evaluate"
                }
            },
            "required": ["conversation_id", "message_content"]
        }

    def __init__(self, adapter: "BasePlatformAdapter" = None, llm_request_factory: Any = None):
        self.adapter = adapter
        self.llm_request_factory = llm_request_factory

    async def execute(self, **kwargs) -> Any:
        conversation_id = kwargs.get("conversation_id")
        message_content = kwargs.get("message_content", "")
        policy = kwargs.get("policy", "evaluate")

        if not message_content:
            return {
                "success": False,
                "error": "缺少 message_content"
            }

        # 根据策略决定是否回复
        should_reply = False
        reply_content = ""

        if policy == "always":
            # 总是回复
            should_reply = True
            reply_content = "我收到了你的消息，但需要更多上下文来给出有意义的回复。"
        elif policy == "random":
            # 随机回复（简化实现）
            import random
            should_reply = random.random() < 0.3  # 30% 概率回复
            if should_reply:
                reply_content = "嗯..." if random.random() < 0.5 else "有意思"
        else:
            # 使用小模型评估
            if self.llm_request_factory:
                try:
                    # 使用 utils_small 模型评估
                    request = self.llm_request_factory.get_request("utils_small")
                    prompt = f"""评估这条消息，判断我是否应该回复。

消息内容：{message_content[:100]}

请只输出 JSON 格式：
{{
    "should_reply": true/false,
    "reason": "原因",
    "interest_level": 0.0-1.0
}}"""

                    content, _ = await request.execute(prompt=prompt)
                    # 简单解析（实际应该用更robust的方法）
                    should_reply = "true" in content.lower()
                    reply_content = "我对你说的这个很感兴趣" if should_reply else ""
                except Exception as e:
                    logger.error(f"小模型评估失败: {e}")
                    # 失败时不回复
                    pass

        if should_reply and reply_content:
            if self.adapter:
                # TODO: 实际发送
                logger.info(f"回复 {conversation_id}: {reply_content}")
            return {
                "success": True,
                "replied": True,
                "reply_content": reply_content,
                "policy": policy
            }

        return {
            "success": True,
            "replied": False,
            "reason": "不需要回复",
            "policy": policy
        }
