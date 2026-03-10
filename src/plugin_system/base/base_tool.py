# src/plugin_system/base/base_tool.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
from .base_plugin import BasePlugin
from .tool_info import ToolInfo
from src.common.logger import get_logger

logger = get_logger("base_tool")


class BaseTool(ABC):
    """
    插件系统中的工具基类。

    特点：
    - 工具元信息由 manifest 中的 ToolInfo 提供
    - 工具执行逻辑使用 async execute()
    - 通过工具名与插件声明进行绑定
    - 可以从插件对象访问配置、自定义资源与日志

    """

    def __init__(self, plugin: BasePlugin | None = None):
        self.plugin = plugin

    # ========== 工具执行接口 ==========
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        工具核心逻辑。

        返回格式须为：
            {"result": ...}          # 成功
            {"error": ...}     # 失败

        （Planner 会根据这两个字段生成日志）
        """
        pass

    def get_config(self, key: str, default=None):
        """获取插件配置值，使用所属的插件的方法。

        Args:
            key: 配置键名，使用嵌套访问如 "section.subsection.key"
            default: 默认值

        Returns:
            Any: 配置值或默认值
        """
        if not self.plugin:
            return default
        return self.plugin.get_config(key, default)
