from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.common.logger import get_logger

from .base_plugin import BasePlugin

logger = get_logger("base_tool")


class BaseTool(ABC):
    """
    插件工具基类。

    工具声明信息由 ToolInfo/manifest 提供，这里只保留执行协议和基础能力。
    """

    def __init__(self, plugin: BasePlugin | None = None):
        self.plugin = plugin

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        执行工具逻辑。

        建议统一返回 ToolResult，同时兼容历史 dict 返回值。
        """
        pass

    def get_config(self, key: str, default=None):
        if not self.plugin:
            return default
        return self.plugin.get_config(key, default)
