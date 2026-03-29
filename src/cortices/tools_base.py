# src/cortices/tools_base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, TYPE_CHECKING, Optional

from src.common.logger import get_logger

if TYPE_CHECKING:
    from src.cortices.manager import CortexManager
    from src.plugin_system.core.plugin_manager import PluginManager

logger = get_logger(__name__)

class BaseTool(ABC):
    """
    工具的抽象基类。
    """
    cortex_manager: Optional[CortexManager]
    plugin_manager: Optional[PluginManager]

    def __init__(self, cortex_manager: Optional[CortexManager] = None, plugin_manager: Optional[PluginManager] = None):
        self.cortex_manager = cortex_manager
        self.plugin_manager = plugin_manager

    @property
    @abstractmethod
    def scope(self) -> List[str]:
        """
        返回工具的作用域列表，例如 ['main'] 或 ['qq_app', 'social']。
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        返回工具的元数据字典。
        应包含: name, description, parameters, required
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        工具的核心逻辑实现。子类应实现此方法。
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """
        生成符合 OpenAI 规范的工具描述。
        """
        meta = self.metadata
        return {
            "type": "function",
            "function": {
                "name": meta["name"],
                "description": meta["description"],
                "parameters": {
                    "type": "object",
                    "properties": meta.get("parameters", {}),
                    "required": meta.get("required", [])
                }
            }
        }