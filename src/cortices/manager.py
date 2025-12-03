import inspect
import json
import os
import importlib
from pathlib import Path
from typing import Any, Dict, List, Type, Optional

from src.common.logger import get_logger
from src.llm_api.dto import ToolCall
from src.cortices.base_cortex import BaseCortex
from src.cortices.tools_base import BaseTool
from src.agent.world_model import WorldModel
from src.system.di.container import container
from src.cortices.cortex_config_loader import load_cortex_config
from pydantic import BaseModel, ValidationError

CORTEX_MANIFEST_FILE = "manifest.json"
logger = get_logger("CortexManager")

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
    _collected_impetus_descriptions: List[str]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CortexManager, cls).__new__(cls)
            cls._instance._tools = {}
            cls._instance._cortices = {}
            cls._instance._collected_impetus_descriptions = []
        return cls._instance


    def _register_tool(self, tool: BaseTool):
        """注册一个工具实例。"""
        tool_name = tool.metadata["name"]
        if tool_name in self._tools:
            logger.warning(f"警告：工具 '{tool_name}' 被重复定义。后一个将覆盖前一个。")
        self._tools[tool_name] = tool
        logger.info(f"工具 '{tool_name}' 已成功注册到作用域 '{tool.scope}'。")

    async def load_all_cortices(self):
        """
        扫描 src/cortices 目录，加载所有 Cortex 模块，并注册它们的工具。
        """
        logger.info("开始加载所有 Cortex...")
        cortices_base_path = Path(os.path.dirname(os.path.abspath(__file__)))
        world_model: WorldModel = container.resolve(WorldModel)

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

                impetus_data = manifest.get("impetus", {})
                impetus_descriptions = impetus_data.get("impetus_description", [])
                self._collected_impetus_descriptions.extend(impetus_descriptions)

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
                        self._register_tool(tool)

                logger.info(f"Cortex '{cortex_name}' 启动成功。")

            except Exception as e:
                logger.error(f"加载 Cortex '{cortex_name}' 失败: {e}", exc_info=True)

        logger.info("所有 Cortex 加载完成。")

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
        self._tools.clear() # 清理工具注册
        logger.info("所有 Cortex 关闭完成。")


    
    def get_tool_schemas(self, scope: str) -> List[Dict]:
        """获取指定作用域下的所有工具 Schema 列表，用于构建 Prompt。"""
        schemas = []
        for tool in self._tools.values():
            if tool.scope == scope:
                schemas.append(tool.get_schema())
        return schemas
    
    def get_collected_impetus_descriptions(self) -> List[str]:
        """获取所有已加载 Cortex 的内在驱动力描述列表。"""
        return self._collected_impetus_descriptions

    async def execute_tool(self, tool_call: ToolCall) -> Any:
        """根据 ToolCall 对象执行相应的工具。"""
        tool_name = tool_call.tool_name
        if tool_name not in self._tools:
            return f"错误：未找到名为 '{tool_name}' 的工具实现。"
        
        tool = self._tools[tool_name]
        args = tool_call.parameters or {}

        try:
            # 假设所有工具的 execute 方法都是异步的
            return await tool.execute(**args)
        except Exception as e:
            logger.error(f"执行工具 '{tool_name}' 时出错: {e}", exc_info=True)
            return f"执行工具 '{tool_name}' 时出错: {e}"