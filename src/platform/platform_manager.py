import asyncio
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Type, Tuple, Optional

from pydantic import BaseModel, ValidationError

from src.common.logger import get_logger
from .platform_base import BasePlatformAdapter, PostMethod
from src.system.di.container import container # 用于解析依赖，例如 EventManager 的 post_method
logger = get_logger("platform manager")


class PlatformManager:
    """
    平台管理器。
    负责动态发现、注册、实例化、运行和管理所有的平台适配器实例。
    这是一个单例。
    """
    _instance: Optional['PlatformManager'] = None
    _is_initialized: bool = False

    def __new__(cls) -> 'PlatformManager':
        if cls._instance is None:
            cls._instance = super(PlatformManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            logger.info("PlatformManager: 初始化...")
            self._registered_adapter_types: Dict[str, Tuple[Type[BasePlatformAdapter], Type[BaseModel]]] = {}
            self._active_adapters: Dict[str, BasePlatformAdapter] = {}
            self._adapter_tasks: Dict[str, asyncio.Task] = {} # 存储每个适配器的主运行任务
            self._is_initialized = True
            self._init_registered_adapter_types()

    def _init_registered_adapter_types(self):
        """
        扫描 src/platform/sources 目录，动态发现并注册所有平台适配器类型及其配置 Schema。
        """
        if self._registered_adapter_types: # 避免重复初始化
            return

        logger.info("PlatformManager: 扫描并注册平台适配器类型...")
        sources_base_path = Path(os.path.dirname(os.path.abspath(__file__))) / "sources"

        for platform_dir in sources_base_path.iterdir():
            # 遍历 sources 目录下的所有文件夹。每个非特殊（不以 __ 开头）的子文件夹名称，例如 qq_napcat 或 wechat，就被视为一个平台类型 (platform_type)。
            if platform_dir.is_dir() and not platform_dir.name.startswith('__'):
                # 获取平台类型标识符
                platform_type = platform_dir.name
                
                try:
                    # 动态导入 Adapter Class
                    adapter_module_path = f"src.platform.sources.{platform_type}.adapter"
                    adapter_module = importlib.import_module(adapter_module_path)
                    adapter_class: Optional[Type[BasePlatformAdapter]] = None
                    for attr_name in dir(adapter_module):
                        attr = getattr(adapter_module, attr_name)
                        if inspect.isclass(attr) and issubclass(attr, BasePlatformAdapter) and attr is not BasePlatformAdapter:
                            adapter_class = attr
                            break
                    if not adapter_class:
                        raise ImportError(f"在模块 '{adapter_module_path}' 中未找到 BasePlatformAdapter 的子类。")
                    
                    # 动态导入 Config Schema
                    config_schema_module_path = f"src.platform.sources.{platform_type}.config_schema"
                    config_schema_module = importlib.import_module(config_schema_module_path)
                    # 约定：Schema 类名为 ConfigSchema
                    config_schema_class_name = "ConfigSchema"
                    config_schema_class: Type[BaseModel] = getattr(config_schema_module, config_schema_class_name)
                    
                    self._register_adapter_type(platform_type, adapter_class, config_schema_class)
                    logger.info(f"成功注册平台类型 '{platform_type}'。")

                except Exception as e:
                    logger.error(f"注册平台类型 '{platform_type}' 失败: {e}", exc_info=True)

    def _register_adapter_type(self, platform_type: str, adapter_class: Type[BasePlatformAdapter], config_schema: Type[BaseModel]):
        """
        注册一个平台适配器类型及其对应的配置 Schema。
        由 PlatformManager 自身在启动时调用

        Args:
            platform_type (str): 平台的类型标识符 (e.g., "qq_napcat")。
            adapter_class (Type[BasePlatformAdapter]): 适配器类的类型。
            config_schema (Type[BaseModel]): 适配器配置的 Pydantic Schema 类。
        """
        if platform_type in self._registered_adapter_types:
            logger.warning(f"平台类型 '{platform_type}' 已注册，将覆盖现有注册。")
        self._registered_adapter_types[platform_type] = (adapter_class, config_schema)

    async def register_and_start(self, adapter_config: BaseModel, post_method: PostMethod) -> BasePlatformAdapter:
        """
        根据提供的配置注册并启动一个平台适配器实例。

        Args:
            adapter_config (BaseModel): 适配器实例的配置对象。此对象必须包含
                                        adapter_id (str) 和 platform_type (str) 字段。
            post_method (PostMethod): 用于将事件提交给主事件管理器的异步函数。

        Returns:
            BasePlatformAdapter: 启动成功的适配器实例。

        Raises:
            ValueError: 如果配置不包含 adapter_id 或 platform_type，
                        或者 platform_type 未注册，或 adapter_id 已存在。
            ValidationError: 如果 adapter_config 与注册的 Schema 不匹配。
            Exception: 启动适配器时发生其他错误。
        """
        if not hasattr(adapter_config, 'adapter_id') or not isinstance(adapter_config.adapter_id, str):
            raise ValueError("适配器配置必须包含一个有效的 'adapter_id' 字符串。")
        if not hasattr(adapter_config, 'platform_type') or not isinstance(adapter_config.platform_type, str):
            raise ValueError("适配器配置必须包含一个有效的 'platform_type' 字符串。")
        
        adapter_id = adapter_config.adapter_id
        platform_type = adapter_config.platform_type

        if adapter_id in self._active_adapters:
            raise ValueError(f"已存在 ID 为 '{adapter_id}' 的活跃适配器。")
        
        registered_info = self._registered_adapter_types.get(platform_type)
        if not registered_info:
            raise ValueError(f"未注册的平台类型: '{platform_type}'。")
        
        adapter_class, config_schema = registered_info

        # 再次验证配置，确保传入的配置符合注册的Schema
        try:
            # 这里的 adapter_config 已经是 BaseModel 实例，直接通过 schema 校验
            # 为了确保类型一致性，可以尝试重新创建实例（但通常没必要，除非是裸字典）
            validated_config = adapter_config
            if not isinstance(validated_config, config_schema):
                 logger.warning(f"传入适配器 '{adapter_id}' 的配置类型与注册Schema不符，尝试重新验证。")
                 validated_config = config_schema.model_validate(adapter_config.model_dump())

        except ValidationError as e:
            raise ValidationError(f"适配器 '{adapter_id}' 配置验证失败: {e}")
        
        # 实例化适配器
        adapter_instance = adapter_class(
            adapter_id=adapter_id,
            platform_type=platform_type,
            config=validated_config, # 传递经过验证的配置对象
            post_method=post_method
        )
        
        try:
            logger.info(f"PlatformManager: 正在启动适配器 '{adapter_id}' ({platform_type})...")
            task = adapter_instance.run()
            self._active_adapters[adapter_id] = adapter_instance
            self._adapter_tasks[adapter_id] = task
            logger.info(f"PlatformManager: 适配器 '{adapter_id}' 已成功启动。")
            return adapter_instance
        except Exception as e:
            logger.error(f"PlatformManager: 启动适配器 '{adapter_id}' 失败: {e}", exc_info=True)
            raise

    async def shutdown_adapter(self, adapter_id: str):
        """
        优雅地停止并移除指定的适配器实例。

        Args:
            adapter_id (str): 要停止的适配器实例的唯一ID。
        
        Raises:
            ValueError: 如果未找到指定ID的活跃适配器。
        """
        adapter = self._active_adapters.get(adapter_id)
        if not adapter:
            raise ValueError(f"未找到 ID 为 '{adapter_id}' 的活跃适配器。")
        
        logger.info(f"PlatformManager: 正在停止适配器 '{adapter_id}'...")
        try:
            await adapter.terminate()
            task = self._adapter_tasks.get(adapter_id)
            if task:
                task.cancel()
                try:
                    await task # 等待任务取消完成
                except asyncio.CancelledError:
                    pass
            
            del self._active_adapters[adapter_id]
            if adapter_id in self._adapter_tasks:
                del self._adapter_tasks[adapter_id]
            logger.info(f"适配器 '{adapter_id}' 已停止。")
        except Exception as e:
            logger.error(f"停止适配器 '{adapter_id}' 失败: {e}", exc_info=True)
            raise

    async def shutdown_all_adapters(self):
        """
        优雅地停止所有正在运行的适配器。
        """
        if not self._active_adapters:
            logger.warning("没有活跃的适配器可供停止。")
            return

        logger.info("正在停止所有平台适配器...")
        # 复制键列表以避免在迭代时修改字典
        adapter_ids_to_shutdown = list(self._active_adapters.keys())
        for adapter_id in adapter_ids_to_shutdown:
            try:
                await self.shutdown_adapter(adapter_id)
            except Exception as e:
                logger.error(f"停止适配器 '{adapter_id}' 失败: {e}")
        logger.info("所有平台适配器已停止。")

    def get_adapter(self, adapter_id: str) -> Optional[BasePlatformAdapter]:
        """
        通过 ID 获取一个已激活的适配器实例。

        Args:
            adapter_id (str): 适配器的唯一 ID。

        Returns:
            Optional[BasePlatformAdapter]: 适配器实例，如果不存在则返回 None。
        """
        return self._active_adapters.get(adapter_id)