"""
MaiBot 插件系统

提供统一的插件开发和管理框架
"""

# 导出主要的公共接口
from .base import (
    BasePlugin,
    BaseTool,
    ConfigField,
    ComponentType,
    ActionActivationType,
    ChatMode,
    ComponentInfo,
    ActionInfo,
    CommandInfo,
    PluginInfo,
    ToolInfo,
    PythonDependency,
    EventHandlerInfo,
    EventType,
)

# 导入工具模块
from .utils import ManifestValidator,PluginConfigManager


__version__ = "0.0.0"

__all__ = [
    # 基础类
    "BasePlugin",
    "BaseTool",
    # 类型定义
    "ComponentType",
    "ActionActivationType",
    "ChatMode",
    "ComponentInfo",
    "ActionInfo",
    "CommandInfo",
    "PluginInfo",
    "ToolInfo",
    "PythonDependency",
    "EventHandlerInfo",
    "EventType",
    # 装饰器
    "ConfigField",
    # 工具函数
    "ManifestValidator",
    "PluginConfigManager"
]
