from __future__ import annotations

from pathlib import Path
import importlib.util
import inspect
from typing import Any, Dict, List, Tuple, Type

from src.common.logger import get_logger
from src.common.tool_registry import ToolDescriptor, ToolRegistry
from src.plugin_system.base import BasePlugin, BaseTool, PluginInfo, ToolInfo
from src.plugin_system.base.parameter_info import ToolParameter
from src.plugin_system.utils import PluginConfigManager

logger = get_logger("plugin_system")


class PluginManager:
    """
    负责插件加载后的实例管理，以及把插件工具注册到统一 ToolRegistry。
    """

    def __init__(self, tool_registry: ToolRegistry):
        self._plugins: Dict[str, BasePlugin] = {}
        self._tools: Dict[str, Tuple[Type[BaseTool], BasePlugin, ToolInfo]] = {}
        self.config_manager = PluginConfigManager()
        self.tool_registry = tool_registry

    def initialize_from_infos(self, plugin_infos: Dict[str, PluginInfo]) -> None:
        logger.info("开始初始化插件系统")

        skip_plugins = set()
        for name, info in plugin_infos.items():
            if not self._check_dependencies(info, plugin_infos):
                logger.error(f"插件 '{name}' 的依赖检查失败，已跳过。")
                skip_plugins.add(name)

        for name, info in plugin_infos.items():
            if name in skip_plugins:
                continue
            try:
                self._initialize_single_plugin(info)
            except Exception as exc:
                logger.error(f"初始化插件 '{name}' 失败: {exc}", exc_info=True)

        logger.info(f"插件初始化完成，共加载 {len(self._plugins)} 个插件。")

    def _initialize_single_plugin(self, info: PluginInfo) -> None:
        plugin_dir = info.metadata.get("plugin_dir")
        if not plugin_dir:
            raise RuntimeError(f"插件 '{info.name}' 缺少 plugin_dir 元数据。")

        plugin_file = Path(plugin_dir) / "plugin.py"
        spec = importlib.util.spec_from_file_location(f"plugin_{info.name}", plugin_file)
        if not spec or not spec.loader:
            raise RuntimeError(f"无法为插件构建导入 spec: {plugin_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        plugin_class = self._find_plugin_class(module)
        if not plugin_class:
            raise RuntimeError(f"插件 '{info.name}' 中未找到 BasePlugin 子类。")

        try:
            config = self.config_manager.load_plugin_config(info)
        except Exception as exc:
            logger.error(f"加载插件 '{info.name}' 配置失败，将使用空配置: {exc}")
            config = {}

        if not self._is_enabled_in_config(config):
            logger.info(f"插件 '{info.name}' 已在配置中禁用。")
            return

        plugin = plugin_class(config=config, plugin_info=info)
        plugin.on_load()

        self._plugins[info.name] = plugin
        logger.info(f"插件 '{info.name}' 已成功加载。")
        self._register_tools(plugin)

    def _register_tools(self, plugin: BasePlugin) -> None:
        tools: List[Tuple[ToolInfo, Type[BaseTool]]] = plugin.get_plugin_tools()

        for tool_info, tool_class in tools:
            name = tool_info.name
            if name in self._tools:
                raise RuntimeError(f"插件工具重复定义: {name}")

            self._tools[name] = (tool_class, plugin, tool_info)
            self.tool_registry.register_tool(
                descriptor=self._tool_info_to_descriptor(tool_info),
                executor=self._build_registry_executor(name),
            )
            logger.info(f"注册插件工具: {name} (plugin={plugin.plugin_name})")

    def _build_registry_executor(self, tool_name: str):
        async def _executor(**kwargs):
            return await self.execute_tool_by_name(tool_name, **kwargs)

        return _executor

    @staticmethod
    def _tool_info_to_descriptor(tool_info: ToolInfo) -> ToolDescriptor:
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in tool_info.tool_parameters:
            if not isinstance(param, ToolParameter):
                raise TypeError("tool_info.tool_parameters 必须由 ToolParameter 构成。")

            schema = {
                "type": param.type,
                "description": param.description,
            }
            if param.choices:
                schema["enum"] = param.choices

            properties[param.name] = schema
            if param.required:
                required.append(param.name)

        return ToolDescriptor(
            name=tool_info.name,
            description=tool_info.description,
            scopes=list(tool_info.scopes or ["global"]),
            parameters=properties,
            required=required,
            source=f"plugin:{tool_info.plugin_name or 'unknown'}",
            metadata=dict(tool_info.metadata or {}),
        )

    def _find_plugin_class(self, module) -> Type[BasePlugin] | None:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                return obj
        return None

    def _check_dependencies(
        self,
        info: PluginInfo,
        plugin_infos: Dict[str, PluginInfo],
    ) -> bool:
        for dep in info.dependencies:
            if dep not in plugin_infos:
                logger.error(f"插件 '{info.name}' 缺少依赖插件: {dep}")
                return False
        return True

    @staticmethod
    def _is_enabled_in_config(config: Dict[str, Any]) -> bool:
        plugin_section = config.get("plugin", {}) if isinstance(config, dict) else {}
        if not isinstance(plugin_section, dict):
            return True

        enabled = plugin_section.get("enabled", True)
        if isinstance(enabled, str):
            return enabled.strip().lower() in {"1", "true", "yes", "on"}
        return bool(enabled)

    def create_tool_instance(self, name: str) -> BaseTool | None:
        entry = self._tools.get(name)
        if not entry:
            return None

        tool_cls, plugin, _tool_info = entry
        try:
            return tool_cls(plugin=plugin)
        except TypeError:
            return tool_cls()

    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            self._tool_info_to_descriptor(tool_info).to_openai_schema()
            for _tool_cls, _plugin, tool_info in self._tools.values()
        ]

    @staticmethod
    def _build_tool_definition(tool_info: ToolInfo) -> Dict[str, Any]:
        return PluginManager._tool_info_to_descriptor(tool_info).to_openai_schema()

    def get_plugin(self, name: str) -> BasePlugin | None:
        return self._plugins.get(name)

    def get_all_tools_with_info(self) -> List[Tuple[Type[BaseTool], BasePlugin, ToolInfo]]:
        return list(self._tools.values())

    def get_callable_tool_schemas_for_scope(self, cortex_scope: str) -> List[Dict[str, Any]]:
        tool_schemas = []
        for _tool_cls, _plugin, tool_info in self._tools.values():
            tool_scopes = tool_info.scopes or ["global"]
            is_hook = any(scope.startswith("hook:") for scope in tool_scopes)
            if is_hook:
                continue

            if "global" in tool_scopes or cortex_scope in tool_scopes:
                tool_schemas.append(self._build_tool_definition(tool_info))

        return tool_schemas

    async def execute_tool_by_name(self, name: str, **kwargs) -> Any:
        instance = self.create_tool_instance(name)
        if not instance:
            logger.error(f"未知的插件工具: {name}")
            return {"error": f"Unknown plugin tool: {name}"}

        try:
            logger.info(f"执行插件工具: {name}, kwargs={kwargs}")
            result = await instance.execute(**kwargs)
            logger.info(f"插件工具完成: {name}, result={result}")
            return result
        except Exception as exc:
            logger.error(f"执行插件工具 '{name}' 时发生错误: {exc}", exc_info=True)
            return {"error": f"Error executing tool '{name}': {str(exc)}"}
