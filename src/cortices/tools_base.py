# src/cortices/tools_base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Coroutine, Callable, TYPE_CHECKING, Optional

from src.common.logger import get_logger

if TYPE_CHECKING:
    from src.cortices.manager import CortexManager
    from src.plugin_system.core.plugin_manager import PluginManager

logger = get_logger(__name__)

class BaseTool(ABC):
    """
    工具的抽象基类。
    每个工具都是一个包含元数据和执行逻辑的可调用对象。
    该基类实现了插件钩子机制，允许在工具执行前后注入插件逻辑。
    """
    cortex_manager: Optional[CortexManager]
    plugin_manager: Optional[PluginManager]

    def __init__(self, cortex_manager: Optional[CortexManager] = None, plugin_manager: Optional[PluginManager] = None):
        self.cortex_manager = cortex_manager
        self.plugin_manager = plugin_manager

    @property
    @abstractmethod
    def cortex_name(self) -> str:
        """
        返回工具所属的Cortex的名称，例如 'qq_chat'。
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        返回工具的元数据字典，用于生成 OpenAI-compatible 的工具 Schema。
        
        应包含:
        - name: str
        - description: str
        - parameters: Dict (JSON Schema)
        - required: List[str]
        """
        pass

    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """
        执行工具的核心逻辑。子类必须实现此方法。
        这是被 `before_run` 和 `after_run` 钩子包裹的内部方法。
        
        使用 **kwargs 接收所有由 LLM 提供的、可能已被 'before_run' 钩子修改过的参数。
        """
        pass

    async def execute(self, **kwargs) -> Any:
        """
        执行工具的公共入口和总控制器。
        它按顺序编排前置钩子、核心逻辑和后置钩子的执行。
        """
        # 1. 前置钩子，可以修改输入参数
        modified_kwargs = await self._execute_hook('before_run', **kwargs)

        # 2. 执行核心逻辑
        result = await self._execute(**modified_kwargs)

        # 3. 后置钩子，可以修改输出结果
        final_result = await self._execute_hook('after_run', result=result)

        return final_result

    async def _execute_hook(self, stage: str, **kwargs) -> Any:
        """
        查找并执行特定阶段（'before_run' 或 'after_run'）的钩子插件。
        """
        if not self.plugin_manager:
            return kwargs if stage == 'before_run' else kwargs.get('result')

        tool_name = self.metadata.get("name", "unknown_tool")
        hook_scope = f"hook:{stage}:{self.cortex_name}.{tool_name}"
        
        # 从 PluginManager 获取所有工具信息，然后自己过滤
        all_plugins_info = self.plugin_manager.get_all_tools_with_info()
        
        hook_plugins = []
        for tool_cls, plugin, tool_info in all_plugins_info:
            if hook_scope in (tool_info.scopes or []):
                hook_plugins.append((tool_cls, plugin, tool_info))
        
        if not hook_plugins:
            return kwargs if stage == 'before_run' else kwargs.get('result')

        logger.info(f"正在为 '{self.cortex_name}.{tool_name}' 执行 '{stage}' 钩子，共 {len(hook_plugins)} 个插件。")

        if stage == 'before_run':
            # 链式处理输入参数
            current_kwargs = kwargs
            for tool_cls, plugin, tool_info in hook_plugins:
                plugin_tool_instance = tool_cls(plugin=plugin)
                # 假设钩子插件的 execute 方法接收 kwargs 并返回修改后的 kwargs
                current_kwargs = await plugin_tool_instance.execute(**current_kwargs)
            return current_kwargs
        
        elif stage == 'after_run':
            # 链式处理结果
            current_result = kwargs.get('result')
            for tool_cls, plugin, tool_info in hook_plugins:
                plugin_tool_instance = tool_cls(plugin=plugin)
                # 假设钩子插件的 execute 方法接收 result 并返回修改后的 result
                current_result = await plugin_tool_instance.execute(result=current_result)
            return current_result
            
        return kwargs if stage == 'before_run' else kwargs.get('result')


    def get_schema(self) -> Dict[str, Any]:
        """
        根据元数据生成完整的 OpenAI-compatible 工具 Schema。
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
