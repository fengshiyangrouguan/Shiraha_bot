# src/cortices/qq_chat/cortex.py
import asyncio
from typing import Any, Dict, Optional, List
import time

from pydantic import BaseModel
from src.common.logger import get_logger
from src.cortex_system.base_cortex import BaseCortex, CortexSignal
from src.cortex_system.tools_base import BaseTool
from .config.config_schema import CortexConfigSchema
from .tools.basic_tools import (
    SendMessageTool,
    GetMessagesTool,
    GetConversationInfoTool,
    QuickReplyTool
)
from .bridge import get_or_create_bridge, clear_bridge

from src.common.di.container import container
from src.core.memory import UnifiedMemory
from src.common.database.database_manager import DatabaseManager

logger = get_logger("qq_chat")


class QQChatCortex(BaseCortex):
    """
    重构后的 QQ 聊天 Cortex

    职责：
    - 纯感知和执行
    - 提供基础工具
    - 上报标准信号
    - 桥接旧的 Event 系统
    """

    def __init__(self):
        super().__init__()
        self.adapter = None
        self.platform_manager = None
        self.database_manager = None
        self.llm_request_factory = None
        self.unified_memory = None
        self.bridge = None

        # 缓存消息列表
        self._message_cache: Dict[str, List[Dict]] = {}
        self.adapter_id = ""

    async def setup(
        self,
        config: BaseModel,
        signal_callback=None,
        skill_manager=None
    ):
        """
        初始化 QQ Chat Cortex
        """
        self.config = config

        # 保存回调
        self._signal_callback = signal_callback

        # 获取依赖
        try:
            from src.platform.platform_manager import PlatformManager
            self.platform_manager = container.resolve(PlatformManager)
        except Exception as e:
            logger.debug(f"PlatformManager 不可用: {e}")
            self.platform_manager = None

        try:
            self.database_manager = container.resolve(DatabaseManager)
        except Exception as e:
            logger.debug(f"DatabaseManager 不可用: {e}")
            self.database_manager = None

        try:
            self.llm_request_factory = container.resolve("LLMRequestFactory")
        except Exception as e:
            logger.debug(f"LLMRequestFactory 不可用: {e}")
            self.llm_request_factory = None

        # 获取统一记忆系统（如果可用）
        try:
            self.unified_memory = container.resolve(UnifiedMemory)
        except Exception as e:
            logger.debug(f"统一记忆系统不可用: {e}")
            self.unified_memory = None

        # 创建桥接器
        self.bridge = get_or_create_bridge(self)
        logger.info("QQ Chat Bridge 已创建")

        # 启动 adapter（如果配置了）
        if hasattr(config, 'adapter') and self.platform_manager:
            try:
                await self._start_adapter(config.adapter, signal_callback)
            except Exception as e:
                logger.error(f"启动 adapter 失败: {e}")

        # 调用父类初始化（会触发能力发现）
        await super().setup(config, signal_callback, skill_manager)

    async def teardown(self):
        """关闭 Cortex"""
        logger.info("正在关闭 QQ Chat Cortex...")

        # 停止 adapter
        if self.adapter_id and self.database_manager:
            try:
                # 这里需要重新注入容器，因为可能状态已变化
                from src.platform.platform_manager import PlatformManager
                platform_manager: PlatformManager = container.resolve(PlatformManager)
                await platform_manager.shutdown_adapter(self.adapter_id)
            except Exception as e:
                logger.error(f"停止 adapter 失败: {e}")
            self.adapter_id = ""

        # 清除桥接器
        clear_bridge(self)

        # 清空消息缓存
        self._message_cache.clear()

        await super().teardown()

    def get_tools(self) -> List[BaseTool]:
        """
        提供基础工具列表

        只包含最基础的工具，无复杂逻辑链
        """
        tools = [
            SendMessageTool(adapter=self.adapter),
            GetMessagesTool(adapter=self.adapter, database_manager=self.database_manager),
            GetConversationInfoTool(adapter=self.adapter, database_manager=self.database_manager),
            QuickReplyTool(adapter=self.adapter, llm_request_factory=self.llm_request_factory),
        ]

        return tools

    async def get_cortex_summary(self) -> str:
        """
        获取当前状态摘要
        """
        conversation_count = len(self._message_cache)
        active_conversations = [
            f"{cid} ({len(msgs)}条消息)"
            for cid, msgs in self._message_cache.items()
            if msgs
        ]

        summary_parts = [
            "QQ Chat Cortex 状态",
            f"- 活跃会话: {conversation_count}",
        ]

        if active_conversations:
            summary_parts.append(f"- 会话详情: {', '.join(active_conversations[:5])}")
            if len(active_conversations) > 5:
                summary_parts.append(f"  (还有 {len(active_conversations) - 5} 个...)")

        # 添加旧系统的状态（如果有）
        if self.database_manager:
            try:
                from src.cortices.qq_chat.data_model.qq_chat_data import QQChatData
                from src.agent.world_model import WorldModel
                world_model = container.resolve(WorldModel)
                qq_chat_data = await world_model.get_cortex_data("qq_chat_data")
                if qq_chat_data:
                    old_summary = await qq_chat_data.get_global_perception_report()
                    if old_summary:
                        summary_parts.append("\n[旧系统状态]")
                        summary_parts.append(old_summary)
            except Exception as e:
                pass

        return "\n".join(summary_parts)

    async def _start_adapter(
        self,
        adapter_config: BaseModel,
        signal_callback=None
    ):
        """
        启动消息适配器
        """
        from src.platform.platform_manager import PlatformManager

        if not self.platform_manager:
            raise RuntimeError("PlatformManager 不可用")

        # 定义信号发送方法
        async def send_signal_to_kernel(event):
            """将事件发送给内核的信号回调"""
            if signal_callback:
                try:
                    await self._handle_event_as_signal(event)
                except Exception as e:
                    logger.error(f"信号处理失败: {e}")

        # 启动 adapter
        self.adapter = await self.platform_manager.register_and_start(
            adapter_config=adapter_config,
            post_method=send_signal_to_kernel
        )

        if self.adapter:
            self.adapter_id = self.adapter.adapter_id
            logger.info(f"QQ Adapter 已启动: {self.adapter_id}")

    async def _handle_event_as_signal(self, event: Any):
        """
        处理事件并转换为信号

        通过桥接器完成转换
        """
        from src.common.event_model.event import Event
        # 如果已经是 Event 对象，直接处理
        # 这里简化处理，直接发出信号
        msg_content = ""

        if hasattr(event, 'event_data') and hasattr(event.event_data, 'LLM_plain_text'):
            msg_content = event.event_data.LLM_plain_text or ""

        user_nickname = ""
        if hasattr(event, 'user_info') and event.user_info:
            user_nickname = event.user_info.user_nickname or ""

        conversation_id = ""
        if hasattr(event, 'conversation_info') and event.conversation_info:
            conversation_id = event.conversation_info.conversation_id or ""

        message_id = ""
        if hasattr(event, 'event_id'):
            message_id = event.event_id

        # 发送消息信号
        self.emit_signal(
            signal_type="message",
            content=f"{user_nickname or '未知用户'}: {msg_content[:100]}{'...' if len(msg_content) > 100 else ''}",
            source_target=conversation_id,
            priority="medium",
            event_id=message_id,
            event_type=event.event_type if hasattr(event, 'event_type') else "unknown",
            tags=list(event.tags) if hasattr(event, 'tags') else []
        )

    async def on_message(
        self,
        conversation_id: str,
        content: str,
        user_id: str = "",
        user_nickname: str = ""
    ):
        """
        接收消息（供外部调用）
        """
        # 添加到缓存
        if conversation_id not in self._message_cache:
            self._message_cache[conversation_id] = []

        message = {
            "content": content,
            "user_id": user_id,
            "user_nickname": user_nickname or user_id,
            "timestamp": time.time()
        }
        self._message_cache[conversation_id].append(message)

        # 记录到记忆系统
        if self.unified_memory:
            from src.core.memory import MemoryType
            import random

            # 随机决定是否记录到长期记忆（模拟人类的"选择性记忆"）
            should_longterm = random.random() < 0.3  # 30% 概率记录

            memory_content = f"[QQ:{conversation_id}] {user_nickname or user_id}: {content}"
            memory_type = MemoryType.LONG_TERM if should_longterm else MemoryType.SHORT_TERM

            await self.unified_memory.store(
                content=memory_content,
                memory_type=memory_type,
                source_cortex="qq",
                source_target=conversation_id,
                tags=["message", "qq", "social" if conversation_id else ""],
                importance=0.3
            )

        # 发送信号
        self.emit_signal(
            signal_type="message",
            content=f"{user_nickname or user_id}: {content[:100]}{'...' if len(content) > 100 else ''}",
            source_target=conversation_id,
            priority="medium"
        )

        logger.debug(f"接收消息: conversation_id={conversation_id}, content={content[:30]}...")

    async def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """获取最近的消息"""
        messages = self._message_cache.get(conversation_id, [])
        return messages[-limit:] if messages else []

    async def clear_message_cache(self, conversation_id: str = ""):
        """清除消息缓存"""
        if conversation_id:
            self._message_cache.pop(conversation_id, None)
        else:
            self._message_cache.clear()

    async def get_conversation_context(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        获取会话上下文（兼容旧系统）

        Returns: 统一格式 {role, content}
        """
        # 1. 从缓存获取
        cached_messages = await self.get_recent_messages(conversation_id, limit)

        # 2. 从桥接器转换旧系统数据
        bridge_context = await self.bridge.extract_context_from_old_system(
            conversation_id=conversation_id,
            target_id=conversation_id
        )

        # 3. 合并并转换为标准格式
        context_messages = []

        # 添加 system 消息
        context_messages.append({
            "role": "system",
            "content": f"这是QQ会话 {conversation_id} 的对话历史。"
        })

        # 添加缓存消息
        for msg in cached_messages:
            context_messages.append({
                "role": "user",
                "content": f"{msg['user_nickname'] or '未知用户'}: {msg['content']}",
                "timestamp": str(msg['timestamp'])
            })

        # 添加桥接的消息
        for msg in bridge_context:
            context_messages.append(msg)

        return context_messages

    async def send_reply(
        self,
        conversation_id: str,
        content: str
    ) -> Dict[str, Any]:
        """
        发送回复

        通过工具或 adapter 发送
        """
        # 尝试通过工具发送
        result = await self.execute_tool("send_message", conversation_id=conversation_id, content=content)

        if result.get("success"):
            # 记录到缓存
            await self.on_message(conversation_id, content, "你自己")

        return result

    async def run_command(
        self,
        command: str,
        **params
    ) -> Dict[str, Any]:
        """
        执行内核指令（通过工具）

        Args:
            command: 指令名称
            **params: 指令参数

        Returns:
            执行结果
        """
        try:
            command = command.lower()

            if command == "send_message":
                return await self.execute_tool("send_message", **params)
            elif command == "get_messages":
                return await self.execute_tool("get_messages", **params)
            elif command == "get_conversation_info":
                return await self.execute_tool("get_conversation_info", **params)
            elif command == "quick_reply":
                return await self.execute_tool("quick_reply", **params)
            else:
                return {
                    "success": False,
                    "error": f"未知命令: {command}"
                }
        except Exception as e:
            logger.error(f"执行命令失败: {command} - {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_cortex_state(self) -> Dict[str, Any]:
        """获取 cortex 状态（用于 Planner 上下文）"""
        return {
            "cortex_name": "qq_chat",
            "active_conversations": len(self._message_cache),
            "conversation_ids": list(self._message_cache.keys()),
            "adapter_id": self.adapter_id,
            "adapter_active": self.adapter is not None
        }
