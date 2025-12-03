# src/plugin_system/base/base_tool.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.plugin_system.base.component_types import ToolInfo, ComponentType
from src.common.logger import get_logger


class BaseTool(ABC):
    """
    插件系统中的工具基类。

    特点：
    - 工具的 *元信息* 使用 class 变量定义（name/description/parameters）
    - 工具执行逻辑使用 async execute()
    - 自动构建 ToolInfo 供插件系统注册
    - 可以从插件对象访问配置、自定义资源与日志

    """

    # 基础元信息（子类必须覆盖）
    name: str = "base_tool"
    """工具的名称"""
    description: str = "工具描述（请在子类中覆盖）"
    """工具的描述"""
    parameters: List[Dict[str, Any]] = []  # 更 Pythonic 的格式
    """ 
工具参数定义（List[Dict]）

每个工具参数应以字典形式描述，其结构通常如下：
{
    "name": str,                     # 参数名称（必须唯一）
    "type": str,           # 参数类型（用于 LLM / planner 的类型检查）
    "description": str,              # 参数功能或用途说明
    "required": bool,                # 是否为必传参数
    "choices": List[str] | None,     # 可选值列表（若需要约束枚举）
}

例如：
parameters = [
    {
        "name": "city",
        "type": "string",
        "description": "要查询天气的城市名称",
        "required": True,
    },
    {
        "name": "unit",
        "type": “string”
        "description": "温度单位",
        "required": False,
        "choices": ["celsius", "fahrenheit"]
    }
]
"""

    def __init__(self):
        self.logger = get_logger(self.name)

    # ========== 参数 / Schema 工具方法 ==========
    @classmethod
    def get_tool_info(cls) -> ToolInfo:
        """
        将工具元信息封装成 ToolInfo（插件注册中心可使用）。
        """
        return ToolInfo(
            name=cls.name,
            component_type=ComponentType.TOOL,
            description=cls.description,
            tool_parameters=cls.parameters

        )

    @classmethod
    def get_definition(cls) -> Dict[str, Any]:
        """
        返回一个符合LLM Tool-Calling API格式的字典。
        """
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param['name']: {
                            'type': param['type'],
                            'description': param['description']
                        } for param in cls.parameters
                    },
                    "required": [param['name'] for param in cls.parameters if param.get('required', False)],
                },
            },
        }

    # ========== 工具执行接口 ==========
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        工具核心逻辑。

        子类只需关心输入参数和执行逻辑，返回 {"name": tool_name, "content": ...} 即可。
        """
        pass

    # ========== 插件辅助工具 ==========
    def log(self, msg: str):
        """工具内部写日志"""
        self.logger.info(f"[{self.name}] {msg}")
