# src/cortices/qq_chat/cortex.py
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.common.di.container import container
from src.common.logger import get_logger
from src.common.database.database_manager import DatabaseManager
from src.core.memory import MemoryType, UnifiedMemory
from src.cortex_system.base_cortex import BaseCortex
from src.cortex_system.tools_base import BaseTool
from src.llm_api.factory import LLMRequestFactory
from src.platform.platform_manager import PlatformManager

from .tools.basic_tools import (
    GetConversationInfoTool,
    GetMessagesTool,
    MuteConversationTool,
    QuickReplyTool,
    SendEmojiTool,
    SendMessageTool,
    ViewConversationListTool,
)

logger = get_logger("qq_chat")


class QQChatCortex(BaseCortex):
    """
    QQ 聊天 Cortex。

    新职责边界：
    1. 只做消息感知、工具执行和信号上报。
    2. 不再承担旧 Event 系统到通用信号的桥接兼容。
    3. 会话上下文统一来自本地缓存与 UnifiedMemory。
    """

    def __init__(self):
        super().__init__()
        self.adapter = None
        self.adapter_id = ""

        self.platform_manager: Optional[PlatformManager] = None
        self.database_manager: Optional[DatabaseManager] = None
        self.llm_request_factory: Optional[LLMRequestFactory] = None
        self.unified_memory: Optional[UnifiedMemory] = None

        # 结构：conversation_id -> [{role, content, ...}]
        self._message_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._muted_conversations: set[str] = set()

    async def setup(self, config: BaseModel, signal_callback=None, skill_manager=None):
        """
        初始化 QQ Cortex。

        这里会注入依赖、启动平台适配器，并把外部消息统一转换为 CortexSignal。
        """
        self.config = config
        self._signal_callback = signal_callback

        try:
            self.platform_manager = container.resolve(PlatformManager)
        except Exception as exc:
            logger.debug(f"PlatformManager 不可用: {exc}")
            self.platform_manager = None

        try:
            self.database_manager = container.resolve(DatabaseManager)
        except Exception as exc:
            logger.debug(f"DatabaseManager 不可用: {exc}")
            self.database_manager = None

        try:
            self.llm_request_factory = container.resolve(LLMRequestFactory)
        except Exception as exc:
            logger.debug(f"LLMRequestFactory 不可用: {exc}")
            self.llm_request_factory = None

        try:
            self.unified_memory = container.resolve(UnifiedMemory)
        except Exception as exc:
            logger.debug(f"UnifiedMemory 不可用: {exc}")
            self.unified_memory = None

        if hasattr(config, "adapter") and self.platform_manager:
            await self._start_adapter(config.adapter)

        await super().setup(config, signal_callback, skill_manager)

    async def teardown(self):
        """关闭 Cortex 并释放适配器资源。"""
        logger.info("正在关闭 QQ Chat Cortex...")

        if self.adapter_id and self.platform_manager:
            try:
                await self.platform_manager.shutdown_adapter(self.adapter_id)
            except Exception as exc:
                logger.error(f"停止 adapter 失败: {exc}")
            self.adapter_id = ""

        self._message_cache.clear()
        await super().teardown()

    def get_tools(self) -> List[BaseTool]:
        """
        暴露给系统的基础 QQ 工具。
        """
        return [
            SendMessageTool(adapter=self.adapter),
            SendEmojiTool(adapter=self.adapter),
            MuteConversationTool(cortex=self),
            GetMessagesTool(adapter=self.adapter, database_manager=self.database_manager, cortex=self),
            GetConversationInfoTool(adapter=self.adapter, database_manager=self.database_manager),
            ViewConversationListTool(cortex=self),
            QuickReplyTool(adapter=self.adapter, llm_request_factory=self.llm_request_factory),
        ]

    async def get_cortex_summary(self) -> str:
        """
        返回当前 QQ 会话状态摘要，供 Planner 直接感知。
        """
        conversation_count = len(self._message_cache)
        summary_parts = [
            "QQ Chat Cortex 状态",
            f"- 活跃会话数：{conversation_count}",
            f"- 适配器状态：{'已连接' if self.adapter else '未连接'}",
        ]

        for conversation_id, messages in list(self._message_cache.items())[:5]:
            summary_parts.append(f"- 会话 {conversation_id}：缓存 {len(messages)} 条消息")

        return "\n".join(summary_parts)

    async def _start_adapter(self, adapter_config: BaseModel):
        """
        启动 QQ 平台适配器，并将平台事件统一送入本 Cortex 的事件处理入口。
        """
        if not self.platform_manager:
            raise RuntimeError("PlatformManager 不可用")

        async def post_method(event):
            await self._handle_platform_event(event)

        self.adapter = await self.platform_manager.register_and_start(
            adapter_config=adapter_config,
            post_method=post_method,
        )

        if self.adapter:
            self.adapter_id = self.adapter.adapter_id
            logger.info(f"QQ Adapter 已启动: {self.adapter_id}")

    async def _handle_platform_event(self, event: Any):
        """
        将平台事件转换为 QQ Cortex 内部的标准消息，再发出内核信号。
        """
        message_payload = self._extract_message_payload(event)
        conversation_id = message_payload["conversation_id"]

        if conversation_id:
            self._append_cache_message(
                conversation_id=conversation_id,
                role="user",
                content=message_payload["content"],
                user_id=message_payload["user_id"],
                user_nickname=message_payload["user_nickname"],
                message_id=message_payload["message_id"],
            )

        await self._store_message_memory(
            conversation_id=conversation_id,
            user_id=message_payload["user_id"],
            user_nickname=message_payload["user_nickname"],
            content=message_payload["content"],
        )

        if conversation_id in self._muted_conversations:
            return

        self.emit_signal(
            signal_type="message",
            content=f"{message_payload['user_nickname'] or message_payload['user_id'] or '未知用户'}: {message_payload['content'][:100]}",
            source_target=conversation_id,
            priority=self._infer_priority(event),
            event_id=message_payload["message_id"],
            event_type=getattr(event, "event_type", "unknown"),
            tags=list(getattr(event, "tags", [])),
            full_content=message_payload["content"],
        )

    def _extract_message_payload(self, event: Any) -> Dict[str, str]:
        """
        从平台事件里提取 QQ 消息所需的核心字段。

        保持这里独立，是为了让主事件处理逻辑更清晰，也便于后续做单元测试。
        """
        content = ""
        user_id = ""
        user_nickname = ""
        conversation_id = ""
        message_id = ""

        if hasattr(event, "event_data") and hasattr(event.event_data, "LLM_plain_text"):
            content = event.event_data.LLM_plain_text or ""

        if hasattr(event, "user_info") and event.user_info:
            user_id = event.user_info.user_id or ""
            user_nickname = event.user_info.user_nickname or ""

        if hasattr(event, "conversation_info") and event.conversation_info:
            conversation_id = event.conversation_info.conversation_id or ""

        if hasattr(event, "event_id"):
            message_id = event.event_id or ""

        return {
            "content": content,
            "user_id": user_id,
            "user_nickname": user_nickname,
            "conversation_id": conversation_id,
            "message_id": message_id,
        }

    def _infer_priority(self, event: Any) -> str:
        """
        依据消息标签推断优先级。
        """
        tags = set(getattr(event, "tags", []) or [])
        if "at_me" in tags or "mentioned_me" in tags:
            return "high"
        return "medium"

    async def on_message(
        self,
        conversation_id: str,
        content: str,
        user_id: str = "",
        user_nickname: str = "",
    ):
        """
        手动接收消息的统一入口，便于测试和外部直接调用。
        """
        self._append_cache_message(
            conversation_id=conversation_id,
            role="user",
            content=content,
            user_id=user_id,
            user_nickname=user_nickname,
        )

        await self._store_message_memory(
            conversation_id=conversation_id,
            user_id=user_id,
            user_nickname=user_nickname,
            content=content,
        )

        self.emit_signal(
            signal_type="message",
            content=f"{user_nickname or user_id or '未知用户'}: {content[:100]}",
            source_target=conversation_id,
            priority="medium",
            full_content=content,
        )

        logger.debug(f"接收消息: conversation_id={conversation_id}, content={content[:30]}...")

    async def get_recent_messages(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """读取指定会话最近缓存的消息。"""
        messages = self._message_cache.get(conversation_id, [])
        return messages[-limit:] if messages else []

    async def clear_message_cache(self, conversation_id: str = ""):
        """清空全部或指定会话的缓存消息。"""
        if conversation_id:
            self._message_cache.pop(conversation_id, None)
        else:
            self._message_cache.clear()

    async def get_conversation_context(self, conversation_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        基于本地缓存与 UnifiedMemory 构建会话上下文。
        """
        context_messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": f"这是 QQ 会话 {conversation_id} 的对话上下文。",
            }
        ]

        cached_messages = await self.get_recent_messages(conversation_id, limit)
        for msg in cached_messages:
            context_messages.append(
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }
            )

        if self.unified_memory:
            try:
                memories = await self.unified_memory.retrieve(
                    query="QQ 对话上下文",
                    source_cortex="qq",
                    source_target=conversation_id,
                    limit=max(1, limit // 2),
                    semantic=False,
                )
                for memory in memories:
                    context_messages.append(
                        {
                            "role": "memory",
                            "content": memory.content,
                        }
                    )
            except Exception as exc:
                logger.warning(f"检索 QQ 会话记忆失败: {exc}")

        return context_messages

    async def send_reply(self, conversation_id: str, content: str) -> Dict[str, Any]:
        """
        发送回复，并在成功后同步写入本地缓存。
        """
        result = await self.execute_tool("send_message", conversation_id=conversation_id, content=content)
        if result.get("success"):
            self._append_cache_message(
                conversation_id=conversation_id,
                role="assistant",
                content=content,
                user_id="self",
                user_nickname="你自己",
            )
        return result

    async def run_command(self, command: str, **params) -> Dict[str, Any]:
        """
        兼容式命令入口，统一转成基础工具调用。
        """
        try:
            command = command.lower()

            if command == "send_message":
                return await self.execute_tool("send_message", **params)
            if command == "get_messages":
                return await self.execute_tool("get_messages", **params)
            if command == "get_conversation_info":
                return await self.execute_tool("get_conversation_info", **params)
            if command == "quick_reply":
                return await self.execute_tool("quick_reply", **params)

            return {"success": False, "error": f"未知命令: {command}"}
        except Exception as exc:
            logger.error(f"执行命令失败: {command} - {exc}")
            return {"success": False, "error": str(exc)}

    def get_cortex_state(self) -> Dict[str, Any]:
        """返回规划器可读取的 Cortex 运行状态。"""
        return {
            "cortex_name": "qq_chat",
            "active_conversations": len(self._message_cache),
            "conversation_ids": list(self._message_cache.keys()),
            "adapter_id": self.adapter_id,
            "adapter_active": self.adapter is not None,
            "muted_conversations": list(self._muted_conversations),
        }

    def mute_conversation(self, conversation_id: str) -> None:
        """将指定会话加入免打扰集合。"""
        self._muted_conversations.add(conversation_id)

    def get_conversation_list_panel(self) -> Dict[str, Any]:
        """
        返回当前会话列表面板。
        """
        conversations = []
        for conversation_id, messages in self._message_cache.items():
            conversations.append(
                {
                    "conversation_id": conversation_id,
                    "cached_messages": len(messages),
                    "muted": conversation_id in self._muted_conversations,
                    "last_message": messages[-1]["content"] if messages else "",
                }
            )

        return {
            "panel": "conversation_list",
            "total": len(conversations),
            "items": conversations,
        }

    def _append_cache_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        user_id: str = "",
        user_nickname: str = "",
        message_id: str = "",
    ) -> None:
        """
        写入本地消息缓存。

        这里统一缓存结构，避免各调用路径在上下文字段上继续分裂。
        """
        if conversation_id not in self._message_cache:
            self._message_cache[conversation_id] = []

        sender_name = user_nickname or user_id or ("你自己" if role == "assistant" else "未知用户")
        self._message_cache[conversation_id].append(
            {
                "role": role,
                "content": f"{sender_name}: {content}",
                "raw_content": content,
                "user_id": user_id,
                "user_nickname": user_nickname,
                "message_id": message_id,
                "timestamp": time.time(),
            }
        )

    async def _store_message_memory(
        self,
        conversation_id: str,
        user_id: str,
        user_nickname: str,
        content: str,
    ) -> None:
        """
        把消息以统一记忆格式写入记忆系统。
        """
        if not self.unified_memory or not content:
            return

        tags = ["message", "qq"]
        if conversation_id:
            tags.append("conversation")

        await self.unified_memory.store(
            content=f"[QQ:{conversation_id}] {user_nickname or user_id or '未知用户'}: {content}",
            memory_type=MemoryType.SHORT_TERM,
            source_cortex="qq",
            source_target=conversation_id,
            tags=tags,
            importance=0.3,
        )
