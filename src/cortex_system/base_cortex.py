"""
Base Cortex - 重构后的基底类

简化 cortex 职责：
1. 纯感知和执行
2. 提供基础工具
3. 上报信号
4. 初始化时引导 agent 测试能力
"""
import asyncio
import inspect
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pydantic import BaseModel
from dataclasses import dataclass

from src.common.logger import get_logger
from src.cortex_system.tools_base import BaseTool

try:
    from src.core.skill import SkillManager
    SKILL_AVAILABLE = True
except ImportError:
    SKILL_AVAILABLE = False

logger = get_logger("new_base_cortex")


@dataclass
class CortexSignal:
    """Cortex 信号"""
    signal_type: str           # 信号类型（message, alert, status_change 等）
    source_cortex: str         # 来源 cortex
    source_target: str         # 来源目标（用户ID、群ID等）
    content: str               # 信号内容
    priority: str = "medium"   # 优先级（low/medium/high/critical）
    timestamp: float = None    # 时间戳
    metadata: Dict[str, Any] = None  # 元数据

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "signal_type": self.signal_type,
            "source_cortex": self.source_cortex,
            "source_target": self.source_target,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def to_context_message(self) -> Dict[str, str]:
        """转换为上下文消息格式 {role, content}"""
        role = "observation"
        source_info = f"[{self.source_cortex}"
        if self.source_target:
            source_info += f":{self.source_target}"
        source_info += "]"

        return {
            "role": role,
            "content": f"{source_info} {self.content}",
            "signal_type": self.signal_type,
            "timestamp": str(self.timestamp)
        }


class BaseCortex(ABC):
    """
    新版 Cortex 基底类

    简化职责：
    - 纯感知和执行，无决策逻辑
    - 提供基础工具
    - 上报标准信号
    - 初始化时引导测试能力
    """

    def __init__(self):
        self._name = self.__class__.__name__
        self._config: Optional[BaseModel] = None
        self._enabled = False

        # 信号回调
        self._signal_callback = None

        # 技能管理器
        self._skill_manager: Optional[SkillManager] = None

        logger.debug(f"{self._name}: 实例化")

    @property
    def cortex_name(self) -> str:
        """Cortex 名称"""
        return self._name.replace("Cortex", "").lower()

    async def setup(self, config: BaseModel, signal_callback=None, skill_manager=None):
        """
        初始化 Cortex

        Args:
            config: 配置对象
            signal_callback: 信号回调函数
            skill_manager: 技能管理器
        """
        self._config = config
        self._signal_callback = signal_callback
        self._skill_manager = skill_manager

        # 检查是否启用
        if hasattr(config, 'enable') and not config.enable:
            self._enabled = False
            logger.info(f"{self._name}: 已禁用")
            return

        self._enabled = True

        # 引导 agent 测试能力
        await self._guide_capability_discovery()

        logger.info(f"{self._name}: 设置完成")

    async def teardown(self):
        """关闭 Cortex"""
        logger.info(f"{self._name}: 已关闭")

    @abstractmethod
    def get_tools(self) -> List["BaseTool"]:
        """
        提供基础工具列表

        每个 cortex 只提供最基础的工具，不包含复杂逻辑链
        """
        pass

    @abstractmethod
    async def get_cortex_summary(self) -> str:
        """
        获取 cortex 当前状态摘要

        用于 Planner 了解 cortex 状态
        """
        return f"{self._name}: 无摘要信息"

    async def _guide_capability_discovery(self):
        """
        引导能力发现

        在 cortex 初始化时，记录能力信息给 agent，
        引导他尝试测试使用这个功能
        """
        if not SKILL_AVAILABLE or not self._skill_manager:
            return

        # 获取基础工具列表
        tools = self.get_tools()
        tool_names = [t.metadata.get("name", t.__class__.__name__) for t in tools]

        # 创建默认 skill 文档
        skill_name = f"{self.cortex_name}_basics"
        description = f"{self._name} 的基础能力：{', '.join(tool_names)}"

        self._skill_manager.create_default_skill(
            cortex=self.cortex_name,
            skill_name=skill_name,
            description=description
        )

        logger.info(f"{self._name}: 已引导能力发现，创建了 {skill_name} skill")

    def emit_signal(
        self,
        signal_type: str,
        content: str,
        source_target: str = "",
        priority: str = "medium",
        **metadata
    ):
        """
        发送信号

        Args:
            signal_type: 信号类型
            content: 信号内容
            source_target: 来源目标
            priority: 优先级
            **metadata: 额外的元数据
        """
        if not self._enabled:
            return

        signal = CortexSignal(
            signal_type=signal_type,
            source_cortex=self.cortex_name,
            source_target=source_target,
            content=content,
            priority=priority,
            metadata=metadata
        )

        # 调用回调
        if self._signal_callback:
            try:
                callback_result = self._signal_callback(signal)
                if inspect.isawaitable(callback_result):
                    # emit_signal 本身是同步方法，因此这里显式调度协程，
                    # 避免 Cortex 在任意线程或同步工具里发信号时丢失 await。
                    asyncio.create_task(callback_result)
            except Exception as e:
                logger.error(f"{self._name}: 信号回调失败: {e}")

        logger.debug(f"{self._name}: 发出信号: {signal_type}")

    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self._enabled

    async def execute_tool(
        self,
        tool_name: str,
        **params
    ) -> Dict[str, Any]:
        """
        执行工具

        Args:
            tool_name: 工具名称
            **params: 工具参数

        Returns:
            执行结果
        """
        tools = self.get_tools()
        tool_map = {t.metadata.get("name", t.__class__.__name__): t for t in tools}

        tool = tool_map.get(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"工具 {tool_name} 不存在"
            }

        try:
            result = await tool.execute(**params)
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"{self._name}: 工具执行失败: {tool_name} - {e}")
            return {
                "success": False,
                "error": str(e)
            }
