from pathlib import Path
import importlib.util
import inspect
from typing import List, Dict, Any, Type, Optional

from .base import BasePlugin, BaseTool
from src.utils.logger import logger

class PluginManager:
    """
    插件管理器，负责加载、注册和访问插件及其工具。
    """

    def __init__(self):
        # manager.py 在 plugin_system，plugins 在同级目录
        base_dir = Path(__file__).resolve().parent.parent  # 上一级目录
        self.plugin_dir: Path = base_dir / "plugins"       # plugins 文件夹路径

        self._plugins: Dict[str, BasePlugin] = {}
        self._tools: Dict[str, Type[BaseTool]] = {}

    def load_plugins(self):
        logger.info(f"开始从 '{self.plugin_dir}' 加载插件...")

        self.plugin_dir.mkdir(exist_ok=True)
        (self.plugin_dir / "__init__.py").touch(exist_ok=True)

        for file in self.plugin_dir.iterdir():
            if file.suffix != ".py" or file.name.startswith("__"):
                continue

            # 使用 importlib.util 从文件路径导入插件
            spec = importlib.util.spec_from_file_location(file.stem, file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, BasePlugin) and obj is not BasePlugin:
                            plugin_instance = obj()
                            plugin_name = plugin_instance.__class__.__name__
                            self._plugins[plugin_name] = plugin_instance
                            logger.info(f"成功加载插件: '{plugin_name}'")
                            self._register_tools(plugin_instance)
                except Exception as e:
                    logger.error(f"加载插件 '{file.stem}' 失败: {e}")

        logger.info(f"插件加载完成，共加载 {len(self._plugins)} 个插件。")

    def _register_tools(self, plugin: BasePlugin):
        try:
            for tool_class in plugin.get_tools():
                if issubclass(tool_class, BaseTool):
                    tool_name = tool_class.name
                    if tool_name in self._tools:
                        logger.warning(f"工具 '{tool_name}' 已存在，将被覆盖。")
                    self._tools[tool_name] = tool_class
                    logger.info(f"  - 注册工具: '{tool_name}'")
        except Exception as e:
            logger.error(f"注册来自 '{plugin.__class__.__name__}' 的工具失败: {e}")

    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        return [tool.get_definition() for tool in self._tools.values()]

    def get_tool(self, name: str) -> Optional[Type[BaseTool]]:
        return self._tools.get(name)
