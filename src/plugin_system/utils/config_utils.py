# src/plugin_system/utils/plugin_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Callable

import datetime
import os
import shutil
import toml

from src.common.logger import get_logger
from src.plugin_system.base.config_types import ConfigField

logger = get_logger("plugin_config")


@dataclass
class PluginConfigManager:
    """负责插件配置文件的生成、加载、版本迁移的工具类"""

    plugin_name: str
    log_prefix: str
    config_schema: Dict[str, Dict[str, ConfigField]]
    config_section_descriptions: Dict[str, str]
    get_manifest_info: Callable[[str, Any], Any]

    # ===== 版本相关 =====
    def get_expected_config_version(self) -> str:
        """从 schema 里拿期望的配置版本号"""
        if "plugin" in self.config_schema and isinstance(self.config_schema["plugin"], dict):
            field = self.config_schema["plugin"].get("version")
            if isinstance(field, ConfigField):
                return field.default
        return "0.0.0"

    @staticmethod
    def get_current_config_version(config: Dict[str, Any]) -> str:
        """从已有配置里拿当前版本号"""
        if "plugin" in config and "version" in config["plugin"]:
            return str(config["plugin"]["version"])
        return "0.0.0"

    # ===== 备份 & 迁移 =====
    def backup_config_file(self, config_file_path: str) -> str:
        """备份配置文件"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{config_file_path}.backup_{timestamp}"

        try:
            shutil.copy2(config_file_path, backup_path)
            logger.info(f"{self.log_prefix} 配置文件已备份到: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"{self.log_prefix} 备份配置文件失败: {e}")
            return ""

    def migrate_config_values(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
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

    # ===== schema -> 默认 config dict =====
    def generate_config_from_schema(self) -> Dict[str, Any]:
        """根据 schema 生成配置数据结构（不写文件）"""
        if not self.config_schema:
            return {}

        config_data: Dict[str, Any] = {}

        for section, fields in self.config_schema.items():
            if isinstance(fields, dict):
                section_data: Dict[str, Any] = {}
                for field_name, field in fields.items():
                    if isinstance(field, ConfigField):
                        section_data[field_name] = field.default
                config_data[section] = section_data

        return config_data

    # ===== TOML 构造 & 写文件 =====
    def build_config_toml(self, config_data: Dict[str, Any] | None, include_version: bool) -> str:
        """根据 schema + config_data 生成 TOML 字符串"""
        toml_str = f"# {self.plugin_name} - 配置文件\n"
        plugin_description = self.get_manifest_info("description", "插件配置文件")
        toml_str += f"# {plugin_description}\n"

        if include_version:
            expected_version = self.get_expected_config_version()
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

    def write_config_file(self, config_file_path: str, config_data: Dict[str, Any] | None = None):
        """写配置文件

        - config_data is None => 生成默认配置（冷启动）
        - config_data is dict  => 写入迁移后的配置（带版本号）
        """
        if config_data is None:
            toml_str = self.build_config_toml(config_data=None, include_version=False)
            msg = "已生成默认配置文件"
        else:
            toml_str = self.build_config_toml(config_data=config_data, include_version=True)
            msg = "配置文件已保存"

        try:
            with open(config_file_path, "w", encoding="utf-8") as f:
                f.write(toml_str)
            logger.info(f"{self.log_prefix} {msg}: {config_file_path}")
        except IOError as e:
            logger.error(f"{self.log_prefix} {msg}失败: {e}", exc_info=True)

    # ===== 顶层：加载 + 迁移 =====
    def load_plugin_config(self, plugin_dir: str, config_file_name: str) -> Dict[str, Any]:
        """对外暴露的入口：加载 + 版本检查 + 自动迁移"""

        if not config_file_name:
            logger.debug(f"{self.log_prefix} 未指定配置文件，跳过加载")
            return {}

        config_file_path = os.path.join(plugin_dir, config_file_name)

        # 如果没有配置文件，尝试生成
        if not os.path.exists(config_file_path):
            logger.info(f"{self.log_prefix} 配置文件不存在，正在生成默认配置...")
            self.write_config_file(config_file_path)

        # 再检查一次，若仍不存在说明写入失败
        if not os.path.exists(config_file_path):
            logger.error(f"{self.log_prefix} 配置文件生成失败，加载被中断。")
            return {}

        file_ext = os.path.splitext(config_file_name)[1].lower()
        if file_ext != ".toml":
            logger.warning(f"{self.log_prefix} 不支持的配置文件格式: {file_ext}，仅支持 .toml")
            return {}

        # 读取现有配置
        with open(config_file_path, "r", encoding="utf-8") as f:
            existing_config = toml.load(f) or {}

        current_version = self.get_current_config_version(existing_config)  # config.toml中版本
        expected_version = self.get_expected_config_version()  # config_schema中版本

        # 当前版本不等于期望版本，即config.toml中版本不等于config_schema中版本，
        if current_version != expected_version:
            logger.info(
                f"{self.log_prefix} 检测到配置版本需要更新: 当前=v{current_version}, 期望=v{expected_version}"
            )
            new_config_structure = self.generate_config_from_schema()
            migrated_config = self.migrate_config_values(existing_config, new_config_structure)
            self.write_config_file(config_file_path, migrated_config)
            logger.info(f"{self.log_prefix} 配置文件已从 v{current_version} 更新到 v{expected_version}")
            return migrated_config

        logger.debug(f"{self.log_prefix} 配置版本匹配 (v{current_version})，直接加载")
        return existing_config
