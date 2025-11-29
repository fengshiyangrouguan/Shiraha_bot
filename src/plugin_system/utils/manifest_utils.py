"""
插件Manifest工具模块

提供manifest文件的验证、生成和管理功能
"""

from typing import Dict, Any
from src.common.logger import get_logger
from src.utils.version_comparator import VersionComparator

logger = get_logger("manifest_utils")


class ManifestValidator:
    """Manifest文件验证器"""

    # 必需字段（必须存在且不能为空）
    REQUIRED_FIELDS = ["manifest_version", "name", "version", "description", "author"]

    # 可选字段（可以不存在或为空）
    OPTIONAL_FIELDS = [
        "license",
        "host_application",
        "homepage_url",
        "repository_url",
        "keywords",
        "categories",
        "default_locale",
        "locales_path",
        "plugin_info",
    ]

    # 建议填写的字段（会给出警告但不会导致验证失败）
    RECOMMENDED_FIELDS = ["license", "keywords", "categories"]

    SUPPORTED_MANIFEST_VERSIONS = [1]

    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []

    def validate_manifest(self, manifest_data: Dict[str, Any]) -> bool:
        """验证manifest数据

        Args:
            manifest_data: manifest数据字典

        Returns:
            bool: 是否验证通过（只有错误会导致验证失败，警告不会）
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()

        # 检查必需字段
        for field in self.REQUIRED_FIELDS:
            if field not in manifest_data:
                self.validation_errors.append(f"缺少必需字段: {field}")
            elif not manifest_data[field]:
                self.validation_errors.append(f"必需字段不能为空: {field}")

        # 检查manifest版本
        if "manifest_version" in manifest_data:
            version = manifest_data["manifest_version"]
            if version not in self.SUPPORTED_MANIFEST_VERSIONS:
                self.validation_errors.append(
                    f"不支持的manifest版本: {version}，支持的版本: {self.SUPPORTED_MANIFEST_VERSIONS}"
                )

        # 检查作者信息格式
        if "author" in manifest_data:
            author = manifest_data["author"]
            if isinstance(author, dict):
                if "name" not in author or not author["name"]:
                    self.validation_errors.append("作者信息缺少name字段或为空")
                # url字段是可选的
                if "url" in author and author["url"]:
                    url = author["url"]
                    if not (url.startswith("http://") or url.startswith("https://")):
                        self.validation_warnings.append("作者URL建议使用完整的URL格式")
            elif isinstance(author, str):
                if not author.strip():
                    self.validation_errors.append("作者信息不能为空")
            else:
                self.validation_errors.append("作者信息格式错误，应为字符串或包含name字段的对象")
        # 检查主机应用版本要求（可选）
        if "host_application" in manifest_data:
            host_app = manifest_data["host_application"]
            if isinstance(host_app, dict):
                min_version = host_app.get("min_version", "")
                max_version = host_app.get("max_version", "")

                # 验证版本字段格式
                for version_field in ["min_version", "max_version"]:
                    if version_field in host_app and not host_app[version_field]:
                        self.validation_warnings.append(f"host_application.{version_field}为空")

                # 检查当前主机版本兼容性
                if min_version or max_version:
                    current_version = VersionComparator.get_current_host_version()
                    is_compatible, error_msg = VersionComparator.is_version_in_range(
                        current_version, min_version, max_version
                    )

                    if not is_compatible:
                        self.validation_errors.append(f"版本兼容性检查失败: {error_msg} (当前版本: {current_version})")
                    else:
                        logger.debug(
                            f"版本兼容性检查通过: 当前版本 {current_version} 符合要求 [{min_version}, {max_version}]"
                        )
            else:
                self.validation_errors.append("host_application格式错误，应为对象")

        # 检查URL格式（可选字段）
        for url_field in ["homepage_url", "repository_url"]:
            if url_field in manifest_data and manifest_data[url_field]:
                url: str = manifest_data[url_field]
                if not (url.startswith("http://") or url.startswith("https://")):
                    self.validation_warnings.append(f"{url_field}建议使用完整的URL格式")

        # 检查数组字段格式（可选字段）
        for list_field in ["keywords", "categories"]:
            if list_field in manifest_data:
                field_value = manifest_data[list_field]
                if field_value is not None and not isinstance(field_value, list):
                    self.validation_errors.append(f"{list_field}应为数组格式")
                elif isinstance(field_value, list):
                    # 检查数组元素是否为字符串
                    for i, item in enumerate(field_value):
                        if not isinstance(item, str):
                            self.validation_warnings.append(f"{list_field}[{i}]应为字符串")

        # 检查建议字段（给出警告）
        for field in self.RECOMMENDED_FIELDS:
            if field not in manifest_data or not manifest_data[field]:
                self.validation_warnings.append(f"建议填写字段: {field}")

        # 检查plugin_info结构（可选）
        if "plugin_info" in manifest_data:
            plugin_info = manifest_data["plugin_info"]
            if isinstance(plugin_info, dict):
                # 检查components数组
                if "components" in plugin_info:
                    components = plugin_info["components"]
                    if not isinstance(components, list):
                        self.validation_errors.append("plugin_info.components应为数组格式")
                    else:
                        for i, component in enumerate(components):
                            if not isinstance(component, dict):
                                self.validation_errors.append(f"plugin_info.components[{i}]应为对象")
                            else:
                                # 检查组件必需字段
                                for comp_field in ["type", "name", "description"]:
                                    if comp_field not in component or not component[comp_field]:
                                        self.validation_errors.append(
                                            f"plugin_info.components[{i}]缺少必需字段: {comp_field}"
                                        )
            else:
                self.validation_errors.append("plugin_info应为对象格式")

        return len(self.validation_errors) == 0

    def get_validation_report(self) -> str:
        """获取验证报告"""
        report = []

        if self.validation_errors:
            report.append("❌ 验证错误:")
            report.extend(f"  - {error}" for error in self.validation_errors)
        if self.validation_warnings:
            report.append("⚠️ 验证警告:")
            report.extend(f"  - {warning}" for warning in self.validation_warnings)
        if not self.validation_errors and not self.validation_warnings:
            report.append("✅ Manifest文件验证通过")

        return "\n".join(report)


# class ManifestGenerator:
#     """Manifest文件生成器"""

#     def __init__(self):
#         self.template = {
#             "manifest_version": 1,
#             "name": "",
#             "version": "1.0.0",
#             "description": "",
#             "author": {"name": "", "url": ""},
#             "license": "MIT",
#             "host_application": {"min_version": "1.0.0", "max_version": "4.0.0"},
#             "homepage_url": "",
#             "repository_url": "",
#             "keywords": [],
#             "categories": [],
#             "default_locale": "zh-CN",
#             "locales_path": "_locales",
#         }

#     def generate_from_plugin(self, plugin_instance: BasePlugin) -> Dict[str, Any]:
#         """从插件实例生成manifest

#         Args:
#             plugin_instance: BasePlugin实例

#         Returns:
#             Dict[str, Any]: 生成的manifest数据
#         """
#         manifest = self.template.copy()

#         # 基本信息
#         manifest["name"] = plugin_instance.plugin_name
#         manifest["version"] = plugin_instance.plugin_version
#         manifest["description"] = plugin_instance.plugin_description

#         # 作者信息
#         if plugin_instance.plugin_author:
#             manifest["author"]["name"] = plugin_instance.plugin_author

#         # 组件信息
#         components = []
#         plugin_components = plugin_instance.get_plugin_components()

#         for component_info, component_class in plugin_components:
#             component_data: Dict[str, Any] = {
#                 "type": component_info.component_type.value,
#                 "name": component_info.name,
#                 "description": component_info.description,
#             }

#             # 添加激活模式信息（对于Action组件）
#             if hasattr(component_class, "focus_activation_type"):
#                 activation_modes = []
#                 if hasattr(component_class, "focus_activation_type"):
#                     activation_modes.append(component_class.focus_activation_type.value)
#                 if hasattr(component_class, "normal_activation_type"):
#                     activation_modes.append(component_class.normal_activation_type.value)
#                 component_data["activation_modes"] = list(set(activation_modes))

#             # 添加关键词信息
#             if hasattr(component_class, "activation_keywords"):
#                 keywords = getattr(component_class, "activation_keywords", [])
#                 if keywords:
#                     component_data["keywords"] = keywords

#             components.append(component_data)

#         manifest["plugin_info"] = {"is_built_in": True, "plugin_type": "general", "components": components}

#         return manifest

#     def save_manifest(self, manifest_data: Dict[str, Any], plugin_dir: str) -> bool:
#         """保存manifest文件

#         Args:
#             manifest_data: manifest数据
#             plugin_dir: 插件目录

#         Returns:
#             bool: 是否保存成功
#         """
#         try:
#             manifest_path = os.path.join(plugin_dir, "_manifest.json")
#             with open(manifest_path, "w", encoding="utf-8") as f:
#                 json.dump(manifest_data, f, ensure_ascii=False, indent=2)
#             logger.info(f"Manifest文件已保存: {manifest_path}")
#             return True
#         except Exception as e:
#             logger.error(f"保存manifest文件失败: {e}")
#             return False


# def validate_plugin_manifest(plugin_dir: str) -> bool:
#     """验证插件目录中的manifest文件

#     Args:
#         plugin_dir: 插件目录路径

#     Returns:
#         bool: 是否验证通过
#     """
#     manifest_path = os.path.join(plugin_dir, "_manifest.json")

#     if not os.path.exists(manifest_path):
#         logger.warning(f"未找到manifest文件: {manifest_path}")
#         return False

#     try:
#         with open(manifest_path, "r", encoding="utf-8") as f:
#             manifest_data = json.load(f)

#         validator = ManifestValidator()
#         is_valid = validator.validate_manifest(manifest_data)

#         logger.info(f"Manifest验证结果:\n{validator.get_validation_report()}")

#         return is_valid

#     except Exception as e:
#         logger.error(f"读取或验证manifest文件失败: {e}")
#         return False


# def generate_plugin_manifest(plugin_instance: BasePlugin, save_to_file: bool = True) -> Optional[Dict[str, Any]]:
#     """为插件生成manifest文件

#     Args:
#         plugin_instance: BasePlugin实例
#         save_to_file: 是否保存到文件

#     Returns:
#         Optional[Dict[str, Any]]: 生成的manifest数据
#     """
#     try:
#         generator = ManifestGenerator()
#         manifest_data = generator.generate_from_plugin(plugin_instance)

#         if save_to_file and plugin_instance.plugin_dir:
#             generator.save_manifest(manifest_data, plugin_instance.plugin_dir)

#         return manifest_data

#     except Exception as e:
#         logger.error(f"生成manifest文件失败: {e}")
#         return None
