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

    # ======== 插件基本信息（子类通过 class 变量覆盖）========
    plugin_name: str = ""  # 插件内部标识符（如 "hello_world_plugin"）
    enable_plugin: bool = True  # 是否启用插件（可以被配置文件覆盖）
    dependencies: List[str] = []  # 依赖的其他插件名称
    python_dependencies: List[PythonDependency] = []  # Python 包依赖
    config_file_name: str = ""  # 配置文件名，如 "config.toml"

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

        # 如果 plugin_name 没写，至少 log_prefix 不至于空
        name_for_log = self.plugin_name or self.__class__.__name__
        self.log_prefix = f"[Plugin:{name_for_log}]"

        # 1. 加载 manifest
        self._load_manifest()

        # 2. 验证插件基本信息
        self._validate_plugin_info()

        # 3. 用工具类加载配置
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

        # 5. 从 config 中更新 enable_plugin（覆盖 class 变量）
        self.enable_plugin = self._config_manager.get_enabled_from_config(
            self.config,
            default_enabled=self.enable_plugin,  # 使用类变量作为默认值
        )

        # 6. 从 manifest 获取显示信息
        self.display_name = self.get_manifest_info("name", self.plugin_name or name_for_log)
        self.plugin_version = self.get_manifest_info("version", "1.0.0")
        self.plugin_description = self.get_manifest_info("description", "")
        self.plugin_author = self._get_author_name()

        # 7. 构造 PluginInfo（供注册中心使用）
        self.plugin_info = PluginInfo(
            name=self.plugin_name,
            display_name=self.display_name,
            description=self.plugin_description,
            version=self.plugin_version,
            author=self.plugin_author,
            enabled=self.enable_plugin,
            is_built_in=False,
            config_file=self.config_file_name or "",
            dependencies=self.dependencies.copy(),
            python_dependencies=self.python_dependencies.copy(),
            # manifest 相关信息
            manifest_data=self.manifest_data.copy(),
            license=self.get_manifest_info("license", ""),
            homepage_url=self.get_manifest_info("homepage_url", ""),
            repository_url=self.get_manifest_info("repository_url", ""),
            keywords=self.get_manifest_info("keywords", []).copy()
            if self.get_manifest_info("keywords") else [],
            categories=self.get_manifest_info("categories", []).copy()
            if self.get_manifest_info("categories") else [],
            min_host_version=self.get_manifest_info("host_application.min_version", ""),
            max_host_version=self.get_manifest_info("host_application.max_version", ""),
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

    # ======== 配置相关 ========
    # def _get_expected_config_version(self) -> str:
    #     """获取插件期望的配置版本号"""
    #     if "plugin" in self.config_schema and isinstance(self.config_schema["plugin"], dict):
    #         config_version_field = self.config_schema["plugin"].get("config_version")
    #         if isinstance(config_version_field, ConfigField):
    #             return config_version_field.default
    #     return "1.0.0"
    #
    # def _get_current_config_version(self, config: Dict[str, Any]) -> str:
    #     """从配置文件中获取当前版本号"""
    #     if "plugin" in config and "config_version" in config["plugin"]:
    #         return str(config["plugin"]["config_version"])
    #     return "0.0.0"
    #
    # def _backup_config_file(self, config_file_path: str) -> str:
    #     """备份配置文件"""
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     backup_path = f"{config_file_path}.backup_{timestamp}"
    #
    #     try:
    #         shutil.copy2(config_file_path, backup_path)
    #         logger.info(f"{self.log_prefix} 配置文件已备份到: {backup_path}")
    #         return backup_path
    #     except Exception as e:
    #         logger.error(f"{self.log_prefix} 备份配置文件失败: {e}")
    #         return ""
    #
    # def _migrate_config_values(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
    #     """将旧配置值迁移到新配置结构中"""
    #
    #     def migrate_section(old_section: Dict[str, Any], new_section: Dict[str, Any], section_name: str) -> Dict[
    #         str, Any]:
    #         result = new_section.copy()
    #
    #         for key, value in old_section.items():
    #             if key in new_section:
    #                 if section_name == "plugin" and key == "config_version":
    #                     logger.debug(
    #                         f"{self.log_prefix} 更新配置版本: {section_name}.{key} = {result[key]} (旧值: {value})"
    #                     )
    #                     continue
    #
    #                 if isinstance(value, dict) and isinstance(new_section[key], dict):
    #                     result[key] = migrate_section(value, new_section[key], f"{section_name}.{key}")
    #                 else:
    #                     result[key] = value
    #                     logger.debug(f"{self.log_prefix} 迁移配置: {section_name}.{key} = {value}")
    #             else:
    #                 logger.warning(f"{self.log_prefix} 配置项 {section_name}.{key} 在新版本中已被移除")
    #
    #         return result
    #
    #     migrated_config: Dict[str, Any] = {}
    #
    #     for section_name, new_section_data in new_config.items():
    #         if (
    #                 section_name in old_config
    #                 and isinstance(old_config[section_name], dict)
    #                 and isinstance(new_section_data, dict)
    #         ):
    #             migrated_config[section_name] = migrate_section(
    #                 old_config[section_name], new_section_data, section_name
    #             )
    #         else:
    #             migrated_config[section_name] = new_section_data
    #             if section_name in old_config:
    #                 logger.warning(f"{self.log_prefix} 配置节 {section_name} 结构已改变，使用默认值")
    #
    #     for section_name in old_config:
    #         if section_name not in migrated_config:
    #             logger.warning(f"{self.log_prefix} 配置节 {section_name} 在新版本中已被移除")
    #
    #     return migrated_config
    #
    # def _generate_config_from_schema(self) -> Dict[str, Any]:
    #     """根据 schema 生成配置数据结构（不写文件）"""
    #     if not self.config_schema:
    #         return {}
    #
    #     config_data: Dict[str, Any] = {}
    #
    #     for section, fields in self.config_schema.items():
    #         if isinstance(fields, dict):
    #             section_data: Dict[str, Any] = {}
    #             for field_name, field in fields.items():
    #                 if isinstance(field, ConfigField):
    #                     section_data[field_name] = field.default
    #             config_data[section] = section_data
    #
    #     return config_data
    #
    # # 统一构造 TOML 字符串的方法，被两个写文件函数复用
    # def _build_config_toml(self, config_data: Dict[str, Any] | None, include_version: bool) -> str:
    #     """根据 schema + config_data 生成 TOML 字符串"""
    #     toml_str = f"# {self.plugin_name} - 配置文件\n"
    #     plugin_description = self.get_manifest_info("description", "插件配置文件")
    #     toml_str += f"# {plugin_description}\n"
    #
    #     if include_version:
    #         expected_version = self._get_expected_config_version()
    #         toml_str += f"# 配置版本: {expected_version}\n"
    #
    #     toml_str += "\n"
    #
    #     for section, fields in self.config_schema.items():
    #         if section in self.config_section_descriptions:
    #             toml_str += f"# {self.config_section_descriptions[section]}\n"
    #
    #         toml_str += f"[{section}]\n\n"
    #
    #         if not isinstance(fields, dict):
    #             continue
    #
    #         # 初次生成时 config_data 可能为 None
    #         section_cfg = (config_data or {}).get(section, {})
    #
    #         for field_name, field in fields.items():
    #             if not isinstance(field, ConfigField):
    #                 continue
    #
    #             toml_str += f"# {field.description}"
    #             if field.required:
    #                 toml_str += " (必需)"
    #             toml_str += "\n"
    #
    #             if field.example:
    #                 toml_str += f"# 示例: {field.example}\n"
    #
    #             if field.choices:
    #                 choices_str = ", ".join(map(str, field.choices))
    #                 toml_str += f"# 可选值: {choices_str}\n"
    #
    #             # 有 config_data 就用迁移后的值，否则用默认值
    #             value = section_cfg.get(field_name, field.default)
    #
    #             if isinstance(value, str):
    #                 toml_str += f'{field_name} = "{value}"\n'
    #             elif isinstance(value, bool):
    #                 toml_str += f"{field_name} = {str(value).lower()}\n"
    #             elif isinstance(value, list):
    #                 if all(isinstance(item, str) for item in value):
    #                     formatted_list = "[" + ", ".join(f'"{item}"' for item in value) + "]"
    #                 else:
    #                     formatted_list = str(value)
    #                 toml_str += f"{field_name} = {formatted_list}\n"
    #             else:
    #                 toml_str += f"{field_name} = {value}\n"
    #
    #             toml_str += "\n"
    #
    #         toml_str += "\n"
    #
    #     return toml_str
    #
    # from typing import Dict, Any
    #
    # def _write_config_file(self, config_file_path: str, config_data: Dict[str, Any] | None = None):
    #     """根据是否传入 config_data 决定写默认配置还是写实际配置
    #
    #     - config_data is None  => 生成默认配置（首次创建）
    #     - config_data is dict  => 保存现有配置（迁移 / 更新后）
    #     """
    #     if not self.config_schema:
    #         logger.debug(f"{self.log_prefix} 插件未定义 config_schema，不生成配置文件")
    #         return
    #
    #     # 根据是否有 config_data 决定是否写版本号
    #     if config_data is None:
    #         # 冷启动：只用 schema 默认值，不带版本信息（或你想要的行为）
    #         toml_str = self._build_config_toml(config_data=None, include_version=False)
    #         msg = "已生成默认配置文件"
    #     else:
    #         # 正常保存：用迁移/更新后的配置 + 带版本号
    #         toml_str = self._build_config_toml(config_data=config_data, include_version=True)
    #         msg = "配置文件已保存"
    #
    #     try:
    #         with open(config_file_path, "w", encoding="utf-8") as f:
    #             f.write(toml_str)
    #         logger.info(f"{self.log_prefix} {msg}: {config_file_path}")
    #     except IOError as e:
    #         logger.error(f"{self.log_prefix} {msg}失败: {e}", exc_info=True)
    #
    # def _load_plugin_config(self):
    #     """加载插件配置文件，支持版本检查和自动迁移"""
    #     if not self.config_file_name:
    #         logger.debug(f"{self.log_prefix} 未指定配置文件，跳过加载")
    #         return
    #
    #     if self.plugin_dir:
    #         plugin_dir = self.plugin_dir
    #     else:
    #         try:
    #             plugin_module_path = inspect.getfile(self.__class__)
    #             plugin_dir = os.path.dirname(plugin_module_path)
    #         except (TypeError, OSError):
    #             module = inspect.getmodule(self.__class__)
    #             if module and hasattr(module, "__file__") and module.__file__:
    #                 plugin_dir = os.path.dirname(module.__file__)
    #             else:
    #                 logger.warning(f"{self.log_prefix} 无法获取插件目录路径，跳过配置加载")
    #                 return
    #
    #     config_file_path = os.path.join(plugin_dir, self.config_file_name)
    #
    #     if not os.path.exists(config_file_path):
    #         logger.info(f"{self.log_prefix} 配置文件 {config_file_path} 不存在，将生成默认配置。")
    #         self._write_config_file(config_file_path)
    #
    #     if not os.path.exists(config_file_path):
    #         logger.warning(f"{self.log_prefix} 配置文件 {config_file_path} 不存在且无法生成。")
    #         return
    #
    #     file_ext = os.path.splitext(self.config_file_name)[1].lower()
    #
    #     if file_ext == ".toml":
    #         with open(config_file_path, "r", encoding="utf-8") as f:
    #             existing_config = toml.load(f) or {}
    #
    #         current_version = self._get_current_config_version(existing_config)
    #
    #         if current_version == "0.0.0":
    #             logger.debug(f"{self.log_prefix} 配置文件无版本信息，跳过版本检查")
    #             self.config = existing_config
    #         else:
    #             expected_version = self._get_expected_config_version()
    #
    #             if current_version != expected_version:
    #                 logger.info(
    #                     f"{self.log_prefix} 检测到配置版本需要更新: 当前=v{current_version}, 期望=v{expected_version}"
    #                 )
    #
    #                 new_config_structure = self._generate_config_from_schema()
    #                 migrated_config = self._migrate_config_values(existing_config, new_config_structure)
    #                 self._write_config_file(config_file_path, migrated_config)
    #
    #                 logger.info(f"{self.log_prefix} 配置文件已从 v{current_version} 更新到 v{expected_version}")
    #                 self.config = migrated_config
    #             else:
    #                 logger.debug(f"{self.log_prefix} 配置版本匹配 (v{current_version})，直接加载")
    #                 self.config = existing_config
    #
    #         logger.debug(f"{self.log_prefix} 配置已从 {config_file_path} 加载")
    #
    #         if "plugin" in self.config and "enabled" in self.config["plugin"]:
    #             # 注意：这里会覆盖 class 变量 enable_plugin
    #             self.enable_plugin = self.config["plugin"]["enabled"]  # type: ignore
    #             logger.debug(f"{self.log_prefix} 从配置更新插件启用状态: {self.enable_plugin}")
    #     else:
    #         logger.warning(f"{self.log_prefix} 不支持的配置文件格式: {file_ext}，仅支持 .toml")
    #         self.config = {}

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
