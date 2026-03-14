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
    支持 AOP（面向切面编程）设计，允许插件在工具执行前后注入逻辑。
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

    async def call_tool(self, **kwargs) -> Any:
        """
        执行工具的公共入口。
        负责执行: before_run 钩子 -> 核心逻辑 -> after_run 钩子
        """
        # 1. 执行前置钩子 (Before Run) - 允许修改或拦截参数
        modified_kwargs = await self._execute_hook('before_run', **kwargs)

        # 2. 执行工具本体逻辑
        # 注意：如果前置钩子返回 None 或特定结构，可以在此处做拦截逻辑
        result = await self.execute(**modified_kwargs)

        # 3. 执行后置钩子 (After Run) - 允许修改输出结果或记录日志
        final_result = await self._execute_hook('after_run', result=result)

        return final_result

    async def _execute_hook(self, stage: str, **kwargs) -> Any:
        """
        从 PluginManager 中查找并链式执行匹配的钩子插件。
        
        匹配规则: 插件的 scope 中包含 "hook:{stage}:{cortex_scope}.{tool_name}"
        """
        if not self.plugin_manager:
            return kwargs if stage == 'before_run' else kwargs.get('result')

        tool_name = self.metadata.get("name", "unknown")
        
        # 遍历当前工具的所有 scope，为每个 scope 尝试匹配钩子
        # 例如工具 scope 为 ['qq_app']，则匹配 'hook:before_run:qq_app.send_msg'
        hook_plugins = []
        all_plugins_info = self.plugin_manager.get_all_tools_with_info()

        for scope_item in self.scope:
            hook_scope_target = f"hook:{stage}:{scope_item}.{tool_name}"
            
            for tool_cls, plugin_instance, tool_info in all_plugins_info:
                if hook_scope_target in (tool_info.scopes or []):
                    hook_plugins.append((tool_cls, plugin_instance))

        if not hook_plugins:
            return kwargs if stage == 'before_run' else kwargs.get('result')

        logger.debug(f"工具 '{tool_name}' 触发阶段 '{stage}' 的钩子插件: {len(hook_plugins)} 个")

        if stage == 'before_run':
            current_kwargs = kwargs
            for tool_cls, plugin in hook_plugins:
                # 实例化插件工具并执行
                instance = tool_cls(plugin=plugin)
                # 钩子插件应接收 kwargs 并返回修改后的 Dict
                current_kwargs = await instance.execute(**current_kwargs)
            return current_kwargs

        elif stage == 'after_run':
            current_result = kwargs.get('result')
            for tool_cls, plugin in hook_plugins:
                instance = tool_cls(plugin=plugin)
                # 钩子插件应接收 result 并返回修改后的 result
                current_result = await instance.execute(result=current_result)
            return current_result

        return kwargs if stage == 'before_run' else kwargs.get('result')

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