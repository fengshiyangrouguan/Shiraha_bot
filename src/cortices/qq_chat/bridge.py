"""
QQ Chat Bridge - 新旧系统桥接层

负责将旧的 qq_chat_data/chat_stream 系统与新的统一系统桥接
"""
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from src.common.logger import get_logger
from src.cortex_system.base_cortex import BaseCortex, CortexSignal

if TYPE_CHECKING:
    from src.cortices.qq_chat.cortex import QQChatCortex
    from src.common.event_model.event import Event

logger = get_logger("qq_chat_bridge")


class QQChatBridge:
    """
    QQ Chat 桥接器

    负责将旧的 Event 系统转换为新的 Cortex 信号，并维护必要的兼容性
    """

    def __init__(self, cortex: "QQChatCortex"):
        self.cortex = cortex
        self._event_queue = []
        self._signal_queue = []

    async def on_event_received(self, event: "Event"):
        """
        当接收到平台事件时调用

        将 Event 转换为 Cortex 信号并发送
        """
        if not self.cortex.is_enabled():
            return

        # 提取关键信息
        conversation_id = ""
        user_nickname = ""
        content = ""
        message_id = event.event_id

        if event.conversation_info:
            conversation_id = event.conversation_info.conversation_id or ""
        if event.user_info:
            user_nickname = event.user_info.user_nickname or ""
        if event.event_data and hasattr(event.event_data, "LLM_plain_text"):
            content = event.event_data.LLM_plain_text or ""

        # 发送消息信号
        self.cortex.emit_signal(
            signal_type="message",
            content=f"{user_nickname or '未知用户'}: {content[:100]}{'...' if len(content) > 100 else ''}",
            source_target=conversation_id,
            priority="medium" if "at_me" in event.tags else "low",
            event_id=message_id,
            full_content=content,
            user_id=event.user_info.user_id if event.user_info else ""
        )

        # 如果是 @ 消息，提高优先级
        if "at_me" in event.tags or "mentioned_me" in event.tags:
            self.cortex.emit_signal(
                signal_type="alert",
                content=f"@被提及: {user_nickname or '未知用户'} 在 {conversation_id} 中@了你",
                source_target=conversation_id,
                priority="high",
                original_event_id=message_id,
                tags=list(event.tags)
            )

    async def message_to_old_system(self, event: "Event"):
        """
        将消息转给旧的 event_processor 处理

        保持向后兼容
        """
        # 注意：这里需要访问事件处理器，但为了避免循环依赖，我们通过回调的方式处理
        event_dict = {
            "type": "qq_event",
            "event": event
        }

        # TODO: 将 event_dict 转发给事件处理器
        logger.debug("Bridge: 消息转发到旧系统")

    async def convert_chat_stream_to_unified_context(
        self,
        conversation_id: str,
        limit: int = 20
    ) -> List[Dict[str, str]]:
        """
        将旧的 chat_stream 转换为统一上下文格式

        Args:
            conversation_id: 会话ID
            limit: 消息数量限制

        Returns:
            统一格式的消息列表
        """
        # TODO: 从 qq_chat_data 获取 chat_stream 并转换
        # 这里需要访问 qq_chat_data，但为了避免循环依赖，我们先返回一个框架

        # 模拟转换逻辑
        pass

        return []

    async def extract_context_from_old_system(
        self,
        conversation_id: str,
        target_id: str,
        task_id: str = ""
    ) -> List[Dict[str, str]]:
        """
        从旧系统提取上下文信息

        用于构建 LLM 调用的上下文
        """
        context_messages = []

        # 创建系统消息
        context_messages.append({
            "role": "system",
            "content": f"这是QQ会话 {conversation_id} 的对话历史。"
        })

        try:
            # 从 cortex 获取 chat_stream
            chat_stream = self._get_chat_stream(conversation_id)
            if chat_stream:
                # 获取最近的消息（已标记为已读的）
                messages = chat_stream.llm_context

                # 添加已读历史分割线
                if messages:
                    context_messages.append({
                        "role": "system",
                        "content": "—— 以上为已回复的历史消息，禁止重复回复 ——"
                    })

                # 添加未回复消息
                unreplied = [m for m in messages if not m.is_replyed]
                is_bot = chat_stream.bot_id

                for msg in unreplied:
                    is_bot_message = is_bot and str(msg.user_id) == str(is_bot)
                    role = "assistant" if is_bot_message else "user"

                    sender_name = msg.user_nickname or "未知用户"
                    content = msg.content or '[空消息]'

                    # 添加消息 ID 引用
                    if msg.message_id:
                        content = f"[消息ID:{msg.message_id}] {content}"

                    context_messages.append({
                        "role": role,
                        "content": f"{sender_name}: {content}",
                        "message_id": msg.message_id
                    })

        except Exception as e:
            logger.error(f"从旧系统提取上下文失败: {e}")

        return context_messages

    def _get_chat_stream(self, conversation_id: str):
        """获取 chat_stream（内部方法）"""
        # 这里需要通过某种方式访问 chat_stream
        # 由于避免循环依赖，我们可以通过 cortex 或直接的数据库访问
        # 简化实现：返回 None，让调用方使用 memory系统
        return None

    def should_trigger_planner(
        self,
        event: "Event"
    ) -> bool:
        """
        判断该事件是否应该触发 Planner

        简单的启发式规则
        """
        # @ 消息始终触发
        if "at_me" in event.tags or "mentioned_me" in event.tags:
            return True

        # 群聊消息一定数量后触发
        if event.conversation_info and event.conversation_info.conversation_type == "group":
            # TODO: 基于聊天流的消息数量判断
            pass

        # 私聊消息直接触发
        if event.conversation_info and event.conversation_info.conversation_type == "private":
            return True

        return False

    def get_attention_priority(
        self,
        event: "Event"
    ) -> str:
        """
        根据事件获取优先级

        low/medium/high/critical
        """
        if "at_me" in event.tags or "mentioned_me" in event.tags:
            return "high"

        if event.conversation_info and event.conversation_info.conversation_type == "group":
            # 群聊为低优先级，除非被@
            return "low"

        return "medium"


# 单例实例（每个 cortex 一个）
_bridge_instances: Dict[str, QQChatBridge] = {}


def get_or_create_bridge(cortex: "QQChatCortex") -> QQChatBridge:
    """获取或创建 cortex 的桥接器实例"""
    cortex_id = id(cortex)

    if cortex_id not in _bridge_instances:
        _bridge_instances[cortex_id] = QQChatBridge(cortex)

    return _bridge_instances[cortex_id]


def clear_bridge(cortex: "QQChatCortex"):
    """清除 cortex 的桥接器实例"""
    cortex_id = id(cortex)
    if cortex_id in _bridge_instances:
        del _bridge_instances[cortex_id]


def clear_all_bridges():
    """清除所有桥接器实例"""
    _bridge_instances.clear()
