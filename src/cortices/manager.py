import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from src.agent.world_model import WorldModel
from src.common.action_model.action_spec import ActionSpec
from src.common.action_model.tool_result import ToolResult
from src.common.di.container import container
from src.common.logger import get_logger
from src.common.tool_registry import ToolDescriptor, ToolRegistry
from src.cortices.base_cortex import BaseCortex
from src.cortices.cortex_config_loader import load_cortex_config
from src.cortices.tools_base import BaseTool
from src.llm_api.dto import ToolCall

CORTEX_MANIFEST_FILE = "manifest.json"
logger = get_logger("cortex")


class CortexManager:
    """
    Cortex 的加载与协调器。

    工具查询与执行统一委托给 ToolRegistry，避免在这里混入插件系统细节。
    """

    _instance = None
    _cortices: Dict[str, BaseCortex]
    _tool_registry: Optional[ToolRegistry]
    _collected_capability_descriptions: List[str]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CortexManager, cls).__new__(cls)
            cls._instance._cortices = {}
            cls._instance._tool_registry = None
            cls._instance._collected_capability_descriptions = []
        return cls._instance

    def _get_tool_registry(self) -> ToolRegistry:
        if self._tool_registry is None:
            self._tool_registry = container.resolve(ToolRegistry)
        return self._tool_registry

    def _register_native_tool(self, tool: BaseTool):
        tool_name = tool.metadata["name"]
        raw_parameters = tool.metadata.get("parameters", {})
        if isinstance(raw_parameters, dict) and raw_parameters.get("type") == "object":
            parameters = raw_parameters.get("properties", {})
            required = raw_parameters.get("required", tool.metadata.get("required", []))
        else:
            parameters = raw_parameters
            required = tool.metadata.get("required", [])

        descriptor = ToolDescriptor(
            name=tool_name,
            description=tool.metadata["description"],
            scopes=list(tool.scope),
            parameters=parameters,
            required=required,
            source=f"cortex:{tool.__class__.__module__}",
        )

        async def _executor(**kwargs):
            return await tool.call_tool(**kwargs)

        self._get_tool_registry().register_tool(descriptor=descriptor, executor=_executor)
        logger.info(f"原生工具 '{tool_name}' 已注册到统一工具表，作用域为 {tool.scope}。")

    async def load_all_cortices(self):
        logger.info("开始加载所有 Cortex。")
        cortices_base_path = Path(os.path.dirname(os.path.abspath(__file__)))
        world_model: WorldModel = container.resolve(WorldModel)
        self._get_tool_registry()

        for cortex_dir in cortices_base_path.iterdir():
            if not cortex_dir.is_dir() or cortex_dir.name.startswith("__"):
                continue

            cortex_name = cortex_dir.name
            manifest_path = cortex_dir / CORTEX_MANIFEST_FILE

            if not manifest_path.exists():
                logger.warning(f"Cortex '{cortex_name}' 缺少 {CORTEX_MANIFEST_FILE}，跳过加载。")
                continue

            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                main_class_path = manifest.get("main_class_path")
                if not main_class_path:
                    logger.warning(f"Cortex '{cortex_name}' 的 manifest 缺少 main_class_path，跳过加载。")
                    continue

                module_path, class_name = main_class_path.rsplit(".", 1)
                validated_config = self._load_and_validate_config(cortex_name, cortex_dir)
                if validated_config is None or (
                    hasattr(validated_config, "enable") and not validated_config.enable
                ):
                    continue

                cortex_module = importlib.import_module(module_path)
                cortex_class: Type[BaseCortex] = getattr(cortex_module, class_name)

                capability_data = manifest.get("capability", {})
                capability_name = capability_data.get("name", cortex_name)
                self._collected_capability_descriptions.append(capability_name + ":")
                self._collected_capability_descriptions.extend(
                    capability_data.get("capability_description", [])
                )
                self._collected_capability_descriptions.append("\n")

                if not (
                    inspect.isclass(cortex_class)
                    and issubclass(cortex_class, BaseCortex)
                    and cortex_class is not BaseCortex
                ):
                    logger.warning(f"'{main_class_path}' 不是有效的 Cortex 子类，跳过加载。")
                    continue

                cortex_instance = cortex_class()
                self._cortices[cortex_name] = cortex_instance
                logger.info(f"发现并实例化 Cortex: '{cortex_name}'")

                await cortex_instance.setup(world_model, validated_config, self)

                if hasattr(cortex_instance, "get_tools") and callable(cortex_instance.get_tools):
                    for tool in cortex_instance.get_tools():
                        self._register_native_tool(tool)

                logger.info(f"Cortex '{cortex_name}' 启动成功。")
            except Exception as exc:
                logger.error(f"加载 Cortex '{cortex_name}' 失败: {exc}", exc_info=True)

        logger.info("所有 Cortex 加载完成。")

    def _load_and_validate_config(self, cortex_name: str, cortex_dir: Path) -> Optional[BaseModel]:
        try:
            config_module_path = f"src.cortices.{cortex_name}.config.config_schema"
            config_schema_module = importlib.import_module(config_module_path)
            cortex_config_schema: Type[BaseModel] = getattr(config_schema_module, "CortexConfigSchema")

            validated_config = load_cortex_config(cortex_dir, cortex_config_schema)
            if hasattr(validated_config, "enable") and not validated_config.enable:
                logger.info(f"Cortex '{cortex_name}' 在配置中被禁用，跳过加载。")
                return None
            return validated_config
        except (ModuleNotFoundError, AttributeError):
            logger.info(f"Cortex '{cortex_name}' 未找到有效配置 Schema，将尝试默认配置。")
            try:
                config_module_path = f"src.cortices.{cortex_name}.config.config_schema"
                config_schema_module = importlib.import_module(config_module_path)
                cortex_config_schema: Type[BaseModel] = getattr(config_schema_module, "CortexConfigSchema")
                validated_config = cortex_config_schema()
                if hasattr(validated_config, "enable") and not validated_config.enable:
                    logger.info(f"Cortex '{cortex_name}' 在默认配置中被禁用，跳过加载。")
                    return None
                return validated_config
            except (ModuleNotFoundError, AttributeError):
                logger.warning(f"Cortex '{cortex_name}' 无法实例化默认配置，将使用空配置。")
                return BaseModel()
            except Exception as exc:
                logger.error(f"实例化 Cortex '{cortex_name}' 默认配置时出错: {exc}")
                return None
        except ValidationError as exc:
            logger.error(f"Cortex '{cortex_name}' 配置校验失败: {exc}，跳过加载。")
            return None
        except Exception as exc:
            logger.error(f"加载 Cortex '{cortex_name}' 配置时发生未知错误: {exc}，跳过加载。")
            return None

    async def shutdown_all_cortices(self):
        logger.info("开始关闭所有 Cortex。")
        for cortex_name, cortex_instance in list(self._cortices.items()):
            try:
                await cortex_instance.teardown()
                logger.info(f"Cortex '{cortex_name}' 关闭成功。")
                del self._cortices[cortex_name]
            except Exception as exc:
                logger.error(f"关闭 Cortex '{cortex_name}' 失败: {exc}", exc_info=True)

        self._get_tool_registry().clear()
        logger.info("所有 Cortex 已关闭，统一工具表已清空。")

    def get_tool_schemas(self, scopes: List[str]) -> List[Dict]:
        return self._get_tool_registry().get_tool_schemas(scopes)

    def get_tool_descriptor(self, tool_name: str) -> Optional[ToolDescriptor]:
        registered = self._get_tool_registry().get_tool(tool_name)
        if not registered:
            return None
        return registered.descriptor

    def get_collected_capability_descriptions(self) -> List[str]:
        return self._collected_capability_descriptions

    @staticmethod
    def _normalize_tool_result(result: Any, tool_name: str) -> ToolResult:
        if isinstance(result, ToolResult):
            return result

        if isinstance(result, dict):
            if "error" in result:
                return ToolResult(
                    success=False,
                    summary=f"工具 '{tool_name}' 执行失败",
                    error_message=str(result["error"]),
                )
            if "result" in result:
                return ToolResult(success=True, summary=str(result["result"]))

        if isinstance(result, str):
            return ToolResult(success=True, summary=result)

        return ToolResult(success=True, summary=str(result))

    async def execute_action(self, action: ActionSpec) -> ToolResult:
        if action.action_type != "tool":
            return ToolResult(
                success=False,
                summary=f"不支持的动作类型: {action.action_type}",
                error_message=f"Unsupported action type: {action.action_type}",
            )

        raw_result = await self._get_tool_registry().execute_tool(
            action.tool_name,
            **(action.parameters or {}),
        )
        return self._normalize_tool_result(raw_result, action.tool_name)

    async def call_tool_by_name(self, tool_name: str, **kwargs) -> ToolResult:
        return await self.execute_action(ActionSpec(tool_name=tool_name, parameters=kwargs))

    async def execute_tool(self, tool_call: ToolCall) -> Any:
        action = ActionSpec(
            tool_name=tool_call.tool_name,
            parameters=tool_call.parameters or {},
        )
        return await self.execute_action(action)
