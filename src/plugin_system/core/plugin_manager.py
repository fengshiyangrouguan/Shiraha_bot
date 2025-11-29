from pathlib import Path
import importlib.util
import inspect
from typing import Dict, Type, List, Tuple, Any

from src.plugin_system.base.base_plugin import BasePlugin
from src.plugin_system.base.base_tool import BaseTool

from src.plugin_system.base.component_types import (
    ComponentType,
    ComponentInfo,
)

from src.utils.logger import logger


class PluginManager:
    """
    插件管理器（支持多组件：Tool / Action / Command / EventHandler）
    插件结构：
        plugins/
            ├─ weather_plugin/
            │      └─ plugin.py
            ├─ hello_plugin/
                   └─ plugin.py
    """

    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.plugin_root: Path = base_dir / "plugins"

        # 插件实例：plugin_id -> BasePlugin instance
        self._plugins: Dict[str, BasePlugin] = {}

        # 组件注册表：不同类型分别维护
        self._tools: Dict[str, Type[BaseTool]] = {}

    # =====主流程=====
    def load_plugins(self):
        print(self.plugin_root)
        logger.info(f"开始从 '{self.plugin_root}' 加载插件...")

        self.plugin_root.mkdir(exist_ok=True)

        for folder in self.plugin_root.iterdir():
            if not folder.is_dir():
                continue

            plugin_file = folder / "plugin.py"
            if not plugin_file.exists():
                logger.debug(f"跳过没有 plugin.py 的目录: {folder.name}")
                continue

            self._load_single_plugin(folder, plugin_file)

        logger.info(f"插件加载完成，共加载 {len(self._plugins)} 个插件。")

    # =====加载单个plugin.py文件=====
    def _load_single_plugin(self, folder: Path, plugin_file: Path):
        spec = importlib.util.spec_from_file_location(f"plugin_{folder.name}", plugin_file)
        if not spec or not spec.loader:
            logger.error(f"无法加载插件文件: {plugin_file}")
            return

        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(f"执行插件模块失败: {plugin_file}, error={e}")
            return

        # 找到 BasePlugin 子类
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                self._register_plugin_folder(obj, folder)
                return

        logger.warning(f"插件目录 '{folder.name}' 中未找到 BasePlugin 子类")

    def _register_plugin_folder(self, plugin_class: Type[BasePlugin], plugin_dir: Path):
        try:
            plugin = plugin_class(plugin_dir=plugin_dir)
        except Exception as e:
            logger.error(f"实例化插件 '{plugin_class.__name__}' 失败: {e}")
            return

        plugin_id = plugin_dir.name
        if plugin.enable_plugin:
            self._plugins[plugin_id] = plugin
            logger.info(f"成功加载插件: {plugin.plugin_name}（ID={plugin_id}）")
        else:
            logger.info(f"{plugin.log_prefix} 插件已禁用，不注册组件。")
        # 注册该插件的所有组件
        self._register_components(plugin)

    # =====注册插件的全部组件=====
    def _register_components(self, plugin: BasePlugin):
        try:
            components: List[Tuple[ComponentInfo, Type]] = plugin.get_plugin_components()
        except AttributeError:
            logger.warning(f"插件 '{plugin.plugin_name}' 未实现 get_components() 方法")
            return

        for comp_info, comp_class in components:
            ctype = comp_info.component_type

            # ============ Tool ============
            if ctype is ComponentType.TOOL:
                self._tools[comp_info.name] = comp_class
                logger.info(f"  注册 Tool: {comp_info.name}")



            else:
                logger.warning(f"未知组件类型: {ctype}")

    # =====对外api=====
    def get_tool(self, name: str):
        return self._tools.get(name)

    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        return [tool.get_definition() for tool in self._tools.values()]
