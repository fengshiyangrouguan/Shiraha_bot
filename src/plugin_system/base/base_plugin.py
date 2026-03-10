# src/plugin_system/base/plugin_base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Type, Any
from .plugin_info import PluginInfo
from .tool_info import ToolInfo
from src.common.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .base_tool import BaseTool

logger = get_logger("base_plugin")


class BasePlugin(ABC):
    """
    所有插件的基类
    """

    def __init__(self, plugin_info: PluginInfo, config: dict | None = None):
        """
        插件从主机接收运行时配置。
        插件不能自己加载文件或环境变量。
        """
        self.plugin_info = plugin_info
        self.config = config or {}

        self.plugin_name = plugin_info.name
        self.log_prefix = f"[Plugin:{self.plugin_name}]"

    # ----------------------------
    # 生命周期
    # ----------------------------

    def on_load(self) -> None:
        """
        插件加载时调用一次。
        """
        pass

    def on_unload(self) -> None:
        """
        插件卸载时调用一次。
        """
        pass

    @abstractmethod
    def get_plugin_tools(
            self,
    ) -> List[Tuple[ToolInfo, Type[BaseTool]]]:
        """
                获取插件包含的工具列表

                Returns:
                    List[tuple[ToolInfo, Type[BaseTool]]]:
                        [(工具信息, 工具类), ...]
                """
        raise NotImplementedError("Subclasses must implement this method")

    # ----------------------------
    # 配置访问（只读）
    # ----------------------------

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取插件配置值，支持嵌套键访问

        Args:
            key: 配置键名，支持嵌套访问如 "section.subsection.key"
            default: 默认值

        Returns:
            Any: 配置值或默认值
        """
        # 支持嵌套键访问
        keys = key.split(".")
        current = self.config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def get_declared_tool_info(self, tool_name: str) -> ToolInfo:
        """从插件声明中读取指定工具的 ToolInfo。"""
        for tool in self.plugin_info.tools:
            if tool.name == tool_name:
                return tool

        raise ValueError(f"插件 '{self.plugin_name}' 未声明工具 '{tool_name}' 的 tool_info")
