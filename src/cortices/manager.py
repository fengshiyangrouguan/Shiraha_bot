import inspect
import json
import os
import importlib
from pathlib import Path
from typing import Any, Dict, List, Type, Optional, TYPE_CHECKING

from src.common.logger import get_logger
from src.llm_api.dto import ToolCall
from src.cortices.base_cortex import BaseCortex
from src.cortices.tools_base import BaseTool
from src.agent.world_model import WorldModel
from src.common.di.container import container
from src.cortices.cortex_config_loader import load_cortex_config
from pydantic import BaseModel, ValidationError
from src.plugin_system.base import ToolInfo
from src.plugin_system.core.plugin_manager import PluginManager

CORTEX_MANIFEST_FILE = "manifest.json"
logger = get_logger("cortex")

class CortexManager:
    """
    Cortex 与工具的管理者。
    负责在启动时发现、加载和配置所有 Cortex。
    从 Cortex 中注册所有工具，并提供执行它们的方法。
    这是一个单例。
    """
    _instance = None
    _tools: Dict[str, BaseTool]
    _cortices: Dict[str, BaseCortex]
    _plugin_manager: Optional[PluginManager]
    _plugin_tools: Dict[str, ToolInfo]
    _collected_capability_descriptions: List[str]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CortexManager, cls).__new__(cls)
            cls._instance._tools = {}
            cls._instance._cortices = {}
            cls._instance._plugin_manager = None
            cls._instance._plugin_tools = {}
            cls._instance._collected_capability_descriptions = []
        return cls._instance

    def _register_native_tool(self, tool: BaseTool):
        """注册一个原生工具实例。"""
        tool_name = tool.metadata["name"]
        if tool_name in self._tools or tool_name in self._plugin_tools:
            logger.warning(f"警告：工具 '{tool_name}' 被重复定义。后一个将覆盖前一个。")
        self._tools[tool_name] = tool
        logger.info(f"原生工具 '{tool_name}' 已成功注册到作用域 '{tool.scope}'。")

    async def load_all_cortices(self):
        """
        扫描 src/cortices 目录，加载所有 Cortex 模块，并注册它们的工具。
        同时加载所有插件工具信息。
        """
        logger.info("开始加载所有 Cortex 及插件工具...")
        cortices_base_path = Path(os.path.dirname(os.path.abspath(__file__)))
        world_model: WorldModel = container.resolve(WorldModel)
        self._plugin_manager = container.resolve(PluginManager)

        # 1. 加载所有原生 Cortex 和它们的工具
        for cortex_dir in cortices_base_path.iterdir():
            if not cortex_dir.is_dir() or cortex_dir.name.startswith('__'):
                continue
            
            cortex_name = cortex_dir.name
            manifest_path = cortex_dir / CORTEX_MANIFEST_FILE
            
            if not manifest_path.exists():
                logger.warning(f"Cortex '{cortex_name}' 缺少 {CORTEX_MANIFEST_FILE}，跳过加载。")
                continue

            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                main_class_path = manifest.get("main_class_path")
                if not main_class_path:
                    logger.warning(f"Cortex '{cortex_name}' 的 {CORTEX_MANIFEST_FILE} 缺少 'main_class_path'，跳过加载。")
                    continue
                
                module_path, class_name = main_class_path.rsplit('.', 1)
                validated_config = self._load_and_validate_config(cortex_name, cortex_dir)
                if validated_config is None or (hasattr(validated_config, 'enable') and not validated_config.enable): continue
                cortex_module = importlib.import_module(module_path)
                cortex_class: Type[BaseCortex] = getattr(cortex_module, class_name)

                # 构建cortex描述列表提供给动机层
                capability_data = manifest.get("capability", {})
                capability_name = capability_data.get("name", cortex_name)
                self._collected_capability_descriptions.append(capability_name+":")
                capability_descriptions = capability_data.get("capability_description", [])
                self._collected_capability_descriptions.extend(capability_descriptions)
                self._collected_capability_descriptions.extend(["\n"]) # 添加分隔符，便于后续构建 Prompt

                validated_config = self._load_and_validate_config(cortex_name, cortex_dir)
                if validated_config is None or (hasattr(validated_config, 'enable') and not validated_config.enable):
                    continue

                cortex_module = importlib.import_module(module_path)
                cortex_class: Type[BaseCortex] = getattr(cortex_module, class_name)

                if not (inspect.isclass(cortex_class) and issubclass(cortex_class, BaseCortex) and cortex_class is not BaseCortex):
                    logger.warning(f"'{main_class_path}' 不是一个有效的 Cortex 子类，跳过加载。")
                    continue

                cortex_instance = cortex_class()
                self._cortices[cortex_name] = cortex_instance

                logger.info(f"发现并实例化 Cortex: '{cortex_name}'")

                # 将 WorldModel、配置和 CortexManager 自身传递给 setup
                await cortex_instance.setup(world_model, validated_config, self) 

                # 注册该 Cortex 提供的工具
                if hasattr(cortex_instance, 'get_tools') and callable(cortex_instance.get_tools):
                    for tool in cortex_instance.get_tools():
                        self._register_native_tool(tool)
                logger.info(f"Cortex '{cortex_name}' 启动成功。")

            except Exception as e:
                logger.error(f"加载 Cortex '{cortex_name}' 失败: {e}", exc_info=True)

        # 2. 加载所有插件工具信息
        if self._plugin_manager:
            all_plugin_tools = self._plugin_manager.get_all_tools_with_info()
            for _tool_cls, _plugin, tool_info in all_plugin_tools:
                is_hook = any(s.startswith("hook:") for s in tool_info.scopes)
                if is_hook:
                    continue
                
                if tool_info.name in self._tools or tool_info.name in self._plugin_tools:
                    logger.warning(f"警告：工具 '{tool_info.name}' 被重复定义。")
                self._plugin_tools[tool_info.name] = tool_info
                logger.info(f"插件工具 '{tool_info.name}' 已成功注册。")


        logger.info("所有 Cortex 和插件工具加载完成。")

    def _load_and_validate_config(self, cortex_name: str, cortex_dir: Path) -> Optional[BaseModel]:
        """加载并验证单个 Cortex 的配置。"""
        try:
            config_module_path = f"src.cortices.{cortex_name}.config.config_schema"
            config_schema_module = importlib.import_module(config_module_path)
            cortex_config_schema: Type[BaseModel] = getattr(config_schema_module, "CortexConfigSchema")
            
            validated_config = load_cortex_config(cortex_dir, cortex_config_schema)
            
            if hasattr(validated_config, 'enable') and not validated_config.enable:
                logger.info(f"Cortex '{cortex_name}' 在配置中被禁用，跳过加载。")
                return None
            return validated_config

        except (ModuleNotFoundError, AttributeError):
            logger.info(f"Cortex '{cortex_name}' 未找到有效的配置 Schema，将使用默认配置。")
            try:
                # 即使没有 schema 文件，也尝试实例化一个空的 Pydantic 模型（如果定义了）
                # 这允许 Cortex 在没有 config.toml 的情况下使用默认值运行
                # 遵循统一命名约定：CortexName/config/config_schema.py:CortexConfigSchema
                config_module_path = f"src.cortices.{cortex_name}.config.config_schema"
                config_schema_module = importlib.import_module(config_module_path)
                cortex_config_schema: Type[BaseModel] = getattr(config_schema_module, "CortexConfigSchema")
                validated_config = cortex_config_schema()
                if hasattr(validated_config, 'enable') and not validated_config.enable:
                    logger.info(f"Cortex '{cortex_name}' 在默认配置中被禁用，跳过加载。")
                    return None
                return validated_config
            except (ModuleNotFoundError, AttributeError):
                 logger.warning(f"Cortex '{cortex_name}' 无法实例化默认配置，将无配置加载。")
                 return BaseModel() # 返回一个空的pydantic基模型
            except Exception as e:
                 logger.error(f"实例化 Cortex '{cortex_name}' 的默认配置时出错: {e}")
                 return None
        except ValidationError as e:
            logger.error(f"Cortex '{cortex_name}' 配置验证失败: {e}，跳过加载。")
            return None
        except Exception as e:
            logger.error(f"加载或验证 Cortex '{cortex_name}' 配置时发生未知错误: {e}，跳过加载。")
            return None

    async def shutdown_all_cortices(self):
        """调用所有已加载 Cortex 的 teardown 方法。"""
        logger.info("开始关闭所有 Cortex...")
        for cortex_name, cortex_instance in list(self._cortices.items()):
            try:
                await cortex_instance.teardown()
                logger.info(f"Cortex '{cortex_name}' 关闭成功。")
                del self._cortices[cortex_name]
            except Exception as e:
                logger.error(f"关闭 Cortex '{cortex_name}' 失败: {e}")
        self._tools.clear()
        self._plugin_tools.clear()
        logger.info("所有 Cortex 及插件工具关闭完成。")

    def get_tool_schemas(self, scopes: List[str]) -> List[Dict]:
        """获取指定作用域下的所有工具 Schema 列表。"""
        if isinstance(scopes, str):
            scopes = [scopes]
        
        schemas = []
        # 1. 查找原生工具
        for tool in self._tools.values(): 
            # 使用 set 的交集判断，效率更高且逻辑清晰
            if set(scopes) & set(tool.scope):
                schemas.append(tool.get_schema())

        # 2. 查找插件工具
        if self._plugin_manager:
            for tool_info in self._plugin_tools.values():
                if set(scopes) & set(tool_info.scopes):
                    schemas.append(self._plugin_manager._build_tool_definition(tool_info))
                
        return schemas
    
    def get_collected_capability_descriptions(self) -> List[str]:
        """获取所有已加载 Cortex 的内在驱动力描述列表。"""
        return self._collected_capability_descriptions

    async def call_tool_by_name(self, tool_name: str, **kwargs) -> Any:
        """
        通过名称和参数调用一个已注册的工具（原生或插件）。
        """
        # 1. 尝试原生工具
        if tool_name in self._tools:
            tool = self._tools[tool_name]
            try:
                return await tool.call_tool(**kwargs)
            except Exception as e:
                logger.error(f"调用原生工具'{tool_name}' 错误: {e}", exc_info=True)
                return f"错误调用工具 '{tool_name}': {e}"

        # 2. Try plugin tool
        elif tool_name in self._plugin_tools and self._plugin_manager:
            return await self._plugin_manager.execute_tool_by_name(tool_name, **kwargs)

        # 3. Not found
        error_msg = f"错误：尝试调用一个未注册的工具 '{tool_name}'。"
        logger.error(error_msg)
        return error_msg

    async def execute_tool(self, tool_call: ToolCall) -> Any:
        """根据 ToolCall 对象执行相应的工具（原生或插件）。"""
        return await self.call_tool_by_name(tool_call.tool_name, **(tool_call.parameters or {}))
