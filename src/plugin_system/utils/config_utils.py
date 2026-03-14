from __future__ import annotations
from typing import Dict, Any, Callable, Tuple, List
import datetime
import os
import shutil
import json
import toml

from src.common.logger import get_logger
from src.plugin_system.base.config_types import ConfigField
from src.plugin_system.base.plugin_info import PluginInfo
from src.common.version_comparator import VersionComparator

logger = get_logger("config_utils")


class ConfigError(ValueError):
    """插件配置相关错误。"""
    pass


class PluginConfigManager:
    """负责插件配置文件的生成、加载、版本迁移的工具类"""

    SCHEMA_TYPE_MAP: Dict[str, type] = {
        "string": str,
        "str": str,
        "boolean": bool,
        "bool": bool,
        "integer": int,
        "int": int,
        "number": float,
        "float": float,
        "array": list,
        "list": list,
        "object": dict,
        "dict": dict,
    }

    def __init__(self) -> None:
        """无参初始化，具体上下文由 load_plugin_config 入口注入。"""
        self.plugin_name: str = ""
        self.log_prefix: str = ""
        self.config_schema: Dict[str, Dict[str, ConfigField]] = {}
        self.config_section_descriptions: Dict[str, str] = {}
        self.get_manifest_info: Callable[[str, Any], Any] = lambda _key, default=None: default

    # ===== schema 处理 =====
    @classmethod
    def _normalize_config_schema(cls, schema: Any) -> Dict[str, Dict[str, ConfigField]]:
        """将 raw schema 规范化为 ConfigField 结构。"""
        if schema is None:
            return {}

        if not isinstance(schema, dict):
            raise ConfigError("config_schema 必须是对象")

        normalized: Dict[str, Dict[str, ConfigField]] = {}

        for section_name, fields in schema.items():
            if not isinstance(section_name, str) or not section_name.strip():
                raise ConfigError("config_schema 的 section 名称必须是非空字符串")

            if not isinstance(fields, dict):
                raise ConfigError(f"config_schema.{section_name} 必须是对象")

            section_schema: Dict[str, ConfigField] = {}

            for field_name, field_meta in fields.items():
                # section 级描述元数据，不参与字段 schema 解析
                if field_name == "$description":
                    continue

                if not isinstance(field_name, str) or not field_name.strip():
                    raise ConfigError(f"config_schema.{section_name} 的字段名必须是非空字符串")

                if isinstance(field_meta, ConfigField):
                    section_schema[field_name] = field_meta
                    continue

                if not isinstance(field_meta, dict):
                    raise ConfigError(f"config_schema.{section_name}.{field_name} 必须是对象")

                raw_type = field_meta.get("type", "string")
                if isinstance(raw_type, type):
                    resolved_type = raw_type
                elif isinstance(raw_type, str):
                    resolved_type = cls.SCHEMA_TYPE_MAP.get(raw_type.strip().lower())
                    if resolved_type is None:
                        raise ConfigError(
                            f"config_schema.{section_name}.{field_name}.type 不支持: {raw_type}"
                        )
                else:
                    raise ConfigError(
                        f"config_schema.{section_name}.{field_name}.type 必须是字符串或类型对象"
                    )

                choices = field_meta.get("choices", [])
                if choices is None:
                    choices = []
                if not isinstance(choices, list):
                    raise ConfigError(f"config_schema.{section_name}.{field_name}.choices 必须是数组或 null")

                description = field_meta.get("description", "")
                if not isinstance(description, str):
                    raise ConfigError(f"config_schema.{section_name}.{field_name}.description 必须是字符串")

                example = field_meta.get("example")
                if example is not None and not isinstance(example, str):
                    example = str(example)

                section_schema[field_name] = ConfigField(
                    type=resolved_type,
                    default=field_meta.get("default"),
                    description=description,
                    example=example,
                    required=bool(field_meta.get("required", False)),
                    choices=choices,
                )

            normalized[section_name] = section_schema

        return normalized

    @staticmethod
    def _extract_section_descriptions(schema: Any) -> Dict[str, str]:
        """从 schema 中提取 section 级描述（$description）。"""
        if not isinstance(schema, dict):
            return {}

        descriptions: Dict[str, str] = {}
        for section_name, section_data in schema.items():
            if not isinstance(section_name, str) or not isinstance(section_data, dict):
                continue

            desc = section_data.get("$description")
            if isinstance(desc, str) and desc.strip():
                descriptions[section_name] = desc.strip()

        return descriptions

    @staticmethod
    def _generate_default_config_dict(config_schema: Dict[str, Dict[str, ConfigField]]) -> Dict[str, Any]:
        """根据 ConfigField schema 生成默认配置字典。"""
        if not config_schema:
            return {}

        config_data: Dict[str, Any] = {}

        for section, fields in config_schema.items():
            if isinstance(fields, dict):
                section_data: Dict[str, Any] = {}
                for field_name, field in fields.items():
                    if isinstance(field, ConfigField):
                        section_data[field_name] = field.default
                config_data[section] = section_data

        return config_data

    @classmethod
    def _load_config_schema_from_file(
        cls,
        plugin_dir: str,
        schema_file_name: str = "_config_schema.json",
    ) -> Tuple[Dict[str, Dict[str, ConfigField]], Dict[str, str]]:
        """从插件目录读取独立 schema json，并规范化为 ConfigField 结构。"""
        schema_path = os.path.join(plugin_dir, schema_file_name)

        if not os.path.exists(schema_path):
            raise ConfigError(f"未找到配置 schema 文件: {schema_path}")

        if os.path.splitext(schema_file_name)[1].lower() != ".json":
            raise ConfigError(f"不支持的 schema 文件格式: {schema_file_name}，仅支持 .json")

        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                raw_schema = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ConfigError(f"读取配置 schema 文件失败: {schema_path} ({e})") from e

        return cls._normalize_config_schema(raw_schema), cls._extract_section_descriptions(raw_schema)

    # ===== 版本相关 =====
    def _get_expected_config_version(self) -> str:
        """从 schema 里拿期望的配置版本号"""
        if "plugin" in self.config_schema and isinstance(self.config_schema["plugin"], dict):
            field = self.config_schema["plugin"].get("version")
            if isinstance(field, ConfigField):
                return field.default
        return "0.0.0"

    @staticmethod
    def _get_current_config_version(config: Dict[str, Any]) -> str:
        """从已有配置里拿当前版本号"""
        if "plugin" in config and "version" in config["plugin"]:
            return str(config["plugin"]["version"])
        return "0.0.0"

    # ===== 备份 & 迁移 =====
    def _backup_config_file(self, config_file_path: str) -> str:
        """备份配置文件"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{config_file_path}.backup_{timestamp}"

        try:
            shutil.copy2(config_file_path, backup_path)
            logger.info(f"{self.log_prefix} 配置文件已备份到: {backup_path}")
            return backup_path
        except OSError as e:
            msg = f"{self.log_prefix} 备份配置文件失败: {e}"
            logger.error(msg, exc_info=True)
            raise ConfigError(msg) from e

    def _migrate_config_values(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """将旧配置值迁移到新配置结构中"""

        def migrate_section(
                old_section: Dict[str, Any], new_section: Dict[str, Any], section_name: str
        ) -> Dict[str, Any]:
            result = new_section.copy()

            for key, value in old_section.items():
                if key in new_section:
                    # version 永远用新的，不迁移旧的
                    if section_name == "plugin" and key == "version":
                        logger.debug(
                            f"{self.log_prefix} 更新配置版本: {section_name}.{key} = {result[key]} (旧值: {value})"
                        )
                        continue

                    if isinstance(value, dict) and isinstance(new_section[key], dict):
                        result[key] = migrate_section(value, new_section[key], f"{section_name}.{key}")
                    else:
                        result[key] = value
                        logger.debug(f"{self.log_prefix} 迁移配置: {section_name}.{key} = {value}")
                else:
                    logger.warning(f"{self.log_prefix} 配置项 {section_name}.{key} 在新版本中已被移除")

            return result

        migrated_config: Dict[str, Any] = {}

        for section_name, new_section_data in new_config.items():
            if (
                    section_name in old_config
                    and isinstance(old_config[section_name], dict)
                    and isinstance(new_section_data, dict)
            ):
                migrated_config[section_name] = migrate_section(
                    old_config[section_name], new_section_data, section_name
                )
            else:
                migrated_config[section_name] = new_section_data
                if section_name in old_config:
                    logger.warning(f"{self.log_prefix} 配置节 {section_name} 结构已改变，使用默认值")

        for section_name in old_config:
            if section_name not in migrated_config:
                logger.warning(f"{self.log_prefix} 配置节 {section_name} 在新版本中已被移除")

        return migrated_config

    # ===== 配置验证 =====
    def _validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证配置字段必需项

        Args:
            config: 已加载的配置字典

        Returns:
            Tuple[bool, List[str]]: (是否通过, 错误信息列表)
        """
        errors = []

        for section, fields in self.config_schema.items():
            if not isinstance(fields, dict):
                continue

            section_config = config.get(section, {})
            if not isinstance(section_config, dict):
                section_config = {}

            for field_name, field in fields.items():
                if not isinstance(field, ConfigField):
                    continue

                if field.required:
                    if field_name not in section_config:
                        errors.append(f"{section}.{field_name}")
                        continue

                    value = section_config.get(field_name)
                    if value is None:
                        errors.append(f"{section}.{field_name}")
                        continue

                    if isinstance(value, str) and not value.strip():
                        errors.append(f"{section}.{field_name}")

        return len(errors) == 0, errors

    # ===== TOML 构造 & 写文件 =====
    def _build_config_toml(self, config_data: Dict[str, Any] | None, include_version: bool) -> str:
        """根据 schema + config_data 生成 TOML 字符串"""
        toml_str = f"# {self.plugin_name} - 配置文件\n"
        plugin_description = self.get_manifest_info("description", "插件配置文件")
        toml_str += f"# {plugin_description}\n"

        if include_version:
            expected_version = self._get_expected_config_version()
            toml_str += f"# 配置版本: {expected_version}\n"

        toml_str += "\n"

        for section, fields in self.config_schema.items():
            if section in self.config_section_descriptions:
                toml_str += f"# {self.config_section_descriptions[section]}\n"

            toml_str += f"[{section}]\n\n"

            if not isinstance(fields, dict):
                toml_str += "\n"
                continue

            section_cfg = (config_data or {}).get(section, {})

            for field_name, field in fields.items():
                if not isinstance(field, ConfigField):
                    continue

                toml_str += f"# {field.description}"
                if field.required:
                    toml_str += " (必需)"
                toml_str += "\n"

                if field.example:
                    toml_str += f"# 示例: {field.example}\n"

                if field.choices:
                    choices_str = ", ".join(map(str, field.choices))
                    toml_str += f"# 可选值: {choices_str}\n"

                value = section_cfg.get(field_name, field.default)

                if isinstance(value, str):
                    toml_str += f'{field_name} = "{value}"\n'
                elif isinstance(value, bool):
                    toml_str += f"{field_name} = {str(value).lower()}\n"
                elif isinstance(value, list):
                    if all(isinstance(item, str) for item in value):
                        formatted_list = "[" + ", ".join(f'"{item}"' for item in value) + "]"
                    else:
                        formatted_list = str(value)
                    toml_str += f"{field_name} = {formatted_list}\n"
                else:
                    toml_str += f"{field_name} = {value}\n"

                toml_str += "\n"

            toml_str += "\n"

        return toml_str

    def _write_config_file(self, config_file_path: str, config_data: Dict[str, Any] | None = None):
        """写配置文件

        - config_data is None => 生成默认配置（冷启动）
        - config_data is dict  => 写入迁移后的配置（带版本号）
        """
        if config_data is None:
            toml_str = self._build_config_toml(config_data=None, include_version=False)
            msg = "已生成默认配置文件"
        else:
            toml_str = self._build_config_toml(config_data=config_data, include_version=True)
            msg = "配置文件已保存"

        try:
            with open(config_file_path, "w", encoding="utf-8") as f:
                f.write(toml_str)
            logger.info(f"{self.log_prefix} {msg}: {config_file_path}")
        except OSError as e:
            err_msg = f"{self.log_prefix} {msg}失败: {e}"
            logger.error(err_msg, exc_info=True)
            raise ConfigError(err_msg) from e

    # ===== 顶层：加载 + 迁移 =====
    def load_plugin_config(
        self,
        plugin_info: PluginInfo,
    ) -> Dict[str, Any]:
        """唯一对外入口：加载 + 版本检查 + 自动迁移。"""

        self.plugin_name = plugin_info.name
        self.log_prefix = f"[Plugin:{plugin_info.name}]"
        self.config_section_descriptions = {}
        self.get_manifest_info = lambda key, default=None: getattr(plugin_info, key, default)

        plugin_dir_obj = plugin_info.metadata.get("plugin_dir")
        if not plugin_dir_obj:
            msg = f"{self.log_prefix} 缺少 plugin_dir，配置加载被中断。"
            logger.error(msg)
            raise ConfigError(msg)

        plugin_dir = str(plugin_dir_obj)
        # 配置 schema 统一从插件目录独立 json 文件加载
        self.config_schema, schema_section_descriptions = self._load_config_schema_from_file(plugin_dir)
        self.config_section_descriptions = schema_section_descriptions
        config_file_name = plugin_info.config_file_name or "config.toml"

        if not config_file_name:
            msg = f"{self.log_prefix} 未指定配置文件名。"
            logger.error(msg)
            raise ConfigError(msg)

        config_file_path = os.path.join(plugin_dir, config_file_name)

        # 如果没有配置文件，尝试生成
        if not os.path.exists(config_file_path):
            logger.info(f"{self.log_prefix} 配置文件不存在，正在生成默认配置...")
            self._write_config_file(config_file_path)

        # 再检查一次，若仍不存在说明写入失败
        if not os.path.exists(config_file_path):
            msg = f"{self.log_prefix} 配置文件生成失败，加载被中断。"
            logger.error(msg)
            raise ConfigError(msg)

        file_ext = os.path.splitext(config_file_name)[1].lower()
        if file_ext != ".toml":
            msg = f"{self.log_prefix} 不支持的配置文件格式: {file_ext}，仅支持 .toml"
            logger.error(msg)
            raise ConfigError(msg)

        # 读取现有配置
        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                existing_config = toml.load(f) or {}
        except (OSError, toml.TomlDecodeError) as e:
            msg = f"{self.log_prefix} 读取配置文件失败: {config_file_path} ({e})"
            logger.error(msg, exc_info=True)
            raise ConfigError(msg) from e

        current_version = self._get_current_config_version(existing_config)  # config.toml中版本
        expected_version = self._get_expected_config_version()  # config_schema中版本

        # 当前版本不等于期望版本，即config.toml中版本不等于config_schema中版本（使用语义版本号比对）
        if VersionComparator.compare_versions(current_version, expected_version) != 0:
            logger.info(
                f"{self.log_prefix} 检测到配置版本需要更新: 当前=v{current_version}, 期望=v{expected_version}"
            )

            self._backup_config_file(config_file_path)
            new_config_structure = self._generate_default_config_dict(self.config_schema)
            migrated_config = self._migrate_config_values(existing_config, new_config_structure)
            self._write_config_file(config_file_path, migrated_config)
            logger.info(f"{self.log_prefix} 配置文件已从 v{current_version} 更新到 v{expected_version}")

            # 验证迁移后的配置
            is_valid, missing_fields = self._validate_config(migrated_config)
            if not is_valid:
                msg = f"{self.log_prefix} 配置校验失败，缺少必需字段: {', '.join(missing_fields)}"
                logger.error(msg)
                raise ConfigError(msg)

            return migrated_config

        logger.debug(f"{self.log_prefix} 配置版本匹配 (v{current_version})，直接加载")

        # 验证配置中的必需项
        is_valid, missing_fields = self._validate_config(existing_config)
        if not is_valid:
            msg = f"{self.log_prefix} 配置校验失败，缺少必需字段: {', '.join(missing_fields)}"
            logger.error(msg)
            raise ConfigError(msg)

        return existing_config
