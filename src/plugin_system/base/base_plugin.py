from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Tuple, Type
import os
import json

from src.common.logger import get_logger
from .plugin_info import PluginInfo, PythonDependency
from .component_types import ToolInfo
from .base_tool import BaseTool
from .config_types import ConfigField
from src.plugin_system.utils.manifest_utils import ManifestValidator
from src.plugin_system.utils.config_utils import PluginConfigManager

logger = get_logger("base_plugin")


class BasePlugin(ABC):
    """
    插件统一基类：
    - 负责：manifest 加载、配置文件加载/迁移、PluginInfo 构造、依赖检查
    - 负责：注册组件（Tool 等）
    子类只需要：
    - 写 class 变量：plugin_name, enable_plugin, dependencies, python_dependencies,
      config_file_name, config_schema
    - 实现 get_plugin_components()
    """

    # ======== 插件基本信息（必须填写） ========
    plugin_name: str = ""  # 插件内部标识符（如 "hello_world_plugin"）
    # ======== 插件基本信息（选择填写） ========
    dependencies: List[str] = []  # 依赖的其他插件名称
    python_dependencies: List[PythonDependency] = []  # Python 包依赖
    config_file_name: str = ""  # 配置文件名，如 "config.toml"
    enable_plugin: bool = True  # 是否启用插件（可以被配置文件覆盖）
    # ======== manifest 相关 ========
    manifest_file_name: str = "_manifest.json"  # manifest 文件名

    # ======== 配置相关（Schema）========
    # 例如：
    # config_schema = {
    #     "plugin": {
    #         "name": ConfigField(...),
    #         "version": ConfigField(...),
    #         "enabled": ConfigField(...),
    #     },
    #     "greeting": {...},
    # }
    config_schema: Dict[str, Dict[str, ConfigField]] = {}
    # 配置节描述（写在生成的 toml 注释里）
    config_section_descriptions: Dict[str, str] = {}

    # ======== 构造函数 ========
    def __init__(self, plugin_dir: str):
        """
        初始化插件

        Args:
            plugin_dir: 插件目录路径，由插件管理器传递
        """
        self.config: Dict[str, Any] = {}  # 当前插件配置
        self.plugin_dir = plugin_dir  # 插件所在目录
        self.manifest_data: Dict[str, Any] = {}  # 当前实例的 manifest 数据

        self.log_prefix = f"[Plugin:{self.plugin_name}]"

        # 1. 加载 manifest
        self._load_manifest()

        # 2. 验证插件基本信息
        self._validate_plugin_info()

        # 3. 加载配置管理器
        self._config_manager = PluginConfigManager(
            plugin_name=self.plugin_name,
            log_prefix=self.log_prefix,
            config_schema=self.config_schema,
            config_section_descriptions=self.config_section_descriptions,
            get_manifest_info=self.get_manifest_info,
        )

        # 4. 加载配置文件
        self.config = self._config_manager.load_plugin_config(
            plugin_dir=self.plugin_dir,
            config_file_name=self.config_file_name,
        )

        # 5. 从 manifest 获取显示信息
        self.display_name = self.get_manifest_info("name", self.plugin_name)
        self.plugin_version = self.get_manifest_info("version", "1.0.0")
        self.plugin_description = self.get_manifest_info("description", "")
        self.plugin_author = self._get_author_name()

        # 6. 构造 PluginInfo（供注册中心使用）
        self.plugin_info = PluginInfo(
            # 插件基本信息
            name=self.plugin_name,
            display_name=self.display_name,
            dependencies=self.dependencies.copy(),
            python_dependencies=self.python_dependencies.copy(),
            # config 相关信息
            version=self.plugin_version,
            enabled=self.config.get("plugin", {}).get("enabled", self.enable_plugin),  # 如果config无enabled属性，则使用基本信息中定义的
            config_file=self.config_file_name or "",
            # manifest 相关信息
            manifest_data=self.manifest_data.copy(),
            author=self.plugin_author,
            description=self.plugin_description,
            license=self.get_manifest_info("license", ""),
            homepage_url=self.get_manifest_info("homepage_url", ""),
            repository_url=self.get_manifest_info("repository_url", ""),
            keywords=self.get_manifest_info("keywords", []).copy()
            if self.get_manifest_info("keywords") else [],
            categories=self.get_manifest_info("categories", []).copy()
            if self.get_manifest_info("categories") else [],
            min_host_version=self.get_manifest_info("host_application.min_version", ""),
            max_host_version=self.get_manifest_info("host_application.max_version", ""),
            is_built_in=False,
        )

        logger.debug(f"{self.log_prefix} 插件基类初始化完成")

    # ======== 子类必须实现：返回组件列表 ========
    @abstractmethod
    def get_plugin_components(
            self,
    ) -> List[
        Union[
            Tuple[ToolInfo, Type[BaseTool]],
            # 未来需要时可以扩展：
            # Tuple[ActionInfo, Type[BaseAction]],
            # Tuple[CommandInfo, Type[BaseCommand]],
            # Tuple[EventHandlerInfo, Type[BaseEventHandler]],
        ]
    ]:
        """
        获取插件包含的组件列表

        Returns:
            List[tuple[ComponentInfo, Type]]:
                [(组件信息, 组件类), ...]
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _validate_plugin_info(self):
        """验证插件基本类变量信息"""
        if not self.plugin_name:
            raise ValueError(f"插件类 {self.__class__.__name__} 必须定义 plugin_name")

        # 验证 manifest 中的必需信息
        if not self.get_manifest_info("name"):
            raise ValueError(f"插件 {self.plugin_name} 的 manifest 中缺少 name 字段")
        if not self.get_manifest_info("description"):
            raise ValueError(f"插件 {self.plugin_name} 的 manifest 中缺少 description 字段")

    # ======== manifest 相关 ========
    def _load_manifest(self):
        """加载 manifest 文件（强制要求）"""
        if not self.plugin_dir:
            raise ValueError(f"{self.log_prefix} 没有插件目录路径，无法加载 manifest")

        manifest_path = os.path.join(self.plugin_dir, self.manifest_file_name)

        if not os.path.exists(manifest_path):
            error_msg = f"{self.log_prefix} 缺少必需的 manifest 文件: {manifest_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest_data = json.load(f)

            logger.debug(f"{self.log_prefix} 成功加载 manifest 文件: {manifest_path}")

            # 验证 manifest 格式
            self._validate_manifest()

        except json.JSONDecodeError as e:
            error_msg = f"{self.log_prefix} manifest 文件格式错误: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except IOError as e:
            error_msg = f"{self.log_prefix} 读取 manifest 文件失败: {e}"
            logger.error(error_msg)
            raise IOError(error_msg)

    def _get_author_name(self) -> str:
        """从 manifest 获取作者名称"""
        author_info = self.get_manifest_info("author", {})
        if isinstance(author_info, dict):
            return author_info.get("name", "")
        else:
            return str(author_info) if author_info else ""

    def _validate_manifest(self):
        """验证 manifest 文件格式（使用强化的验证器）"""
        if not self.manifest_data:
            raise ValueError(f"{self.log_prefix} manifest 数据为空，验证失败")

        validator = ManifestValidator()
        is_valid = validator.validate_manifest(self.manifest_data)

        # 记录验证结果
        if validator.validation_errors or validator.validation_warnings:
            report = validator.get_validation_report()
            logger.info(f"{self.log_prefix} Manifest 验证结果:\n{report}")

        # 如果有验证错误，抛出异常
        if not is_valid:
            error_msg = f"{self.log_prefix} Manifest 文件验证失败"
            if validator.validation_errors:
                error_msg += f": {'; '.join(validator.validation_errors)}"
            raise ValueError(error_msg)

    def get_manifest_info(self, key: str, default: Any = None) -> Any:
        """获取 manifest 信息，支持点分割嵌套键，如 'author.name'"""
        if not self.manifest_data:
            return default

        keys = key.split(".")
        value: Any = self.manifest_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    # ======== 依赖检查 & 配置访问 ========
    def _check_dependencies(self) -> bool:
        """检查插件依赖（TODO: 接入 component_registry 做真正的依赖检查）"""
        # from src.plugin_system.core.component_registry import component_registry
        #
        # if not self.dependencies:
        #     return True
        #
        # for dep in self.dependencies:
        #     if not component_registry.get_plugin_info(dep):
        #         logger.error(f"{self.log_prefix} 缺少依赖插件: {dep}")
        #         return False
        return True

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取插件配置值，支持 'section.key' 形式嵌套访问"""
        keys = key.split(".")
        current: Any = self.config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current
