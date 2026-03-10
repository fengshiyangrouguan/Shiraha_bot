from pathlib import Path
import importlib.util
import inspect
from typing import Dict, Type, List, Tuple, Any

from src.plugin_system.base import BasePlugin, BaseTool, ToolInfo, PluginInfo
from src.plugin_system.base.parameter_info import ToolParameter
from src.plugin_system.utils import PluginConfigManager
from src.utils.logger import logger

'''TODO:增加工具卸载功能，涉及调用插件的 on_unload 生命周期方法'''

class PluginManager:
    """
    插件 Manager
    功能: 管理插件的注册、初始化和工具注册
    """

    def __init__(self):
        # plugin_name -> BasePlugin instance
        self._plugins: Dict[str, BasePlugin] = {}

        # tool_name -> (Tool class, owning plugin instance, ToolInfo)
        # 保存插件引用以便 later 实例化时传递配置
        self._tools: Dict[str, Tuple[Type[BaseTool], BasePlugin, ToolInfo]] = {}
        self.config_manager = PluginConfigManager()

    # ========================
    # 主入口
    # ========================

    def initialize_from_infos(
        self,
        plugin_infos: Dict[str, PluginInfo],
    ) -> None:
        """
        根据 PluginInfo 初始化插件

        Args:
            plugin_infos: plugin_name -> PluginInfo
        """
        logger.info("开始根据插件声明初始化插件")

        # 1. 先做声明级依赖检查，并记录需要跳过的插件
        skip_plugins = set()
        for name, info in plugin_infos.items():
            if not self._check_dependencies(info, plugin_infos):
                logger.error(f"插件 '{name}' 依赖不满足，跳过初始化")
                skip_plugins.add(name)

        # 2. 初始化插件
        for name, info in plugin_infos.items():
            if name in skip_plugins:
                continue
            try:
                self._initialize_single_plugin(info)
            except Exception as e:
                logger.error(f"初始化插件 '{name}' 失败: {e}")

        logger.info(f"插件初始化完成，共加载 {len(self._plugins)} 个插件")

    # ========================
    # 单插件初始化
    # ========================

    def _initialize_single_plugin(
        self,
        info: PluginInfo,
    ) -> None:
        """
        初始化单个插件

        Args:
            info: PluginInfo
        """
        plugin_dir = info.metadata.get("plugin_dir")
        if not plugin_dir:
            msg = f"插件 '{info.name}' 缺少 plugin_dir 信息"
            logger.error(msg)
            raise RuntimeError(msg)

        logger.debug(f"初始化插件: {info.name}")

        plugin_file = Path(plugin_dir) / "plugin.py"

        spec = importlib.util.spec_from_file_location(
            f"plugin_{info.name}", plugin_file
        )
        if not spec or not spec.loader:
            msg = f"无法创建插件模块 spec: {plugin_file}"
            logger.error(msg)
            raise RuntimeError(msg)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        plugin_class = self._find_plugin_class(module)
        if not plugin_class:
            msg = f"插件 '{info.name}' 中未找到 BasePlugin 子类"
            logger.error(msg)
            raise RuntimeError(msg)

        try:
            config = self.config_manager.load_plugin_config(info)
        except Exception as e:
            logger.error(f"插件 '{info.name}' 配置加载失败，使用空配置: {e}")
            config = {}

        if not self._is_enabled_in_config(config):
            logger.info(f"插件 '{info.name}' 在配置中被禁用，跳过加载")
            return

        plugin = plugin_class(
            config=config,
            plugin_info=info,
        )

        # 生命周期：on_load
        plugin.on_load()

        self._plugins[info.name] = plugin
        logger.info(f"插件 '{info.name}' 已加载")

        # 注册工具
        self._register_tools(plugin)

    # ========================
    # 工具注册
    # ========================

    def _register_tools(self, plugin: BasePlugin) -> None:
        tools: List[Tuple[ToolInfo, Type[BaseTool]]] = plugin.get_plugin_tools()

        for tool_info, tool_class in tools:
            name = tool_info.name
            if name in self._tools:
                msg = f"Tool 名称冲突: {name}"
                logger.error(msg)
                raise RuntimeError(msg)

            # 记录提供工具的插件，以便 later 创建实例时使用其 config
            self._tools[name] = (tool_class, plugin, tool_info)
            logger.info(f"  注册 Tool: {name} (来自插件 {plugin.plugin_name})")

    # ========================
    # 辅助方法
    # ========================

    def _find_plugin_class(self, module) -> Type[BasePlugin] | None:
        """在模块中查找 BasePlugin 子类"""
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                return obj
        return None

    def _check_dependencies(
        self,
        info: PluginInfo,
        plugin_infos: Dict[str, PluginInfo],
    ) -> bool:
        """声明级依赖检查"""
        for dep in info.dependencies:
            if dep not in plugin_infos:
                logger.error(f"插件 '{info.name}' 缺少依赖插件: {dep}")
                return False
        return True

    @staticmethod
    def _is_enabled_in_config(config: Dict[str, Any]) -> bool:
        """从配置中读取 plugin.enabled，默认启用。"""
        plugin_section = config.get("plugin", {}) if isinstance(config, dict) else {}
        if not isinstance(plugin_section, dict):
            return True

        enabled = plugin_section.get("enabled", True)
        if isinstance(enabled, str):
            return enabled.strip().lower() in {"1", "true", "yes", "on"}
        return bool(enabled)

    # ========================
    # 对外 API
    # ========================

    def create_tool_instance(self, name: str) -> BaseTool | None:
        """
        返回工具实例。
        """
        entry = self._tools.get(name)
        if not entry:
            return None
        tool_cls, plugin, tool_info = entry
        try:
            return tool_cls(plugin=plugin)
        except TypeError:
            return tool_cls()

    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        # 工具声明来自 manifest 解析出的 ToolInfo
        return [self._build_tool_definition(tool_info) for _tool_cls, _plugin, tool_info in self._tools.values()]

    @staticmethod
    def _build_tool_definition(tool_info: ToolInfo) -> Dict[str, Any]:
        """将 ToolInfo 转成 LLM Tool-Calling 所需的定义结构。"""
        if not isinstance(tool_info, ToolInfo):
            raise TypeError("tool_info 必须是 ToolInfo")

        if not tool_info.name or not tool_info.description:
            raise ValueError("tool_info.name 和 tool_info.description 不能为空")

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in tool_info.tool_parameters:
            if not isinstance(param, ToolParameter):
                raise TypeError("tool_info.tool_parameters 列表必须由 ToolParameter 构成")

            schema = {
                "type": param.type,
                "description": param.description,
            }
            if param.choices:
                schema["enum"] = param.choices

            properties[param.name] = schema
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": tool_info.name,
                "description": tool_info.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def get_plugin(self, name: str) -> BasePlugin | None:
        return self._plugins.get(name)
    
