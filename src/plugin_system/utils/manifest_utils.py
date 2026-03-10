"""
插件Manifest工具模块

提供manifest文件的验证、管理功能
"""
from pathlib import Path
from typing import Any, Dict
import json
from src.common.logger import get_logger
from src.plugin_system.base.parameter_info import ToolParameter
from src.plugin_system.base.tool_info import ToolInfo
from src.plugin_system.base.plugin_info import PluginInfo, PythonDependency
from src.utils.version_comparator import VersionComparator

logger = get_logger("manifest_utils")


class ManifestError(ValueError):
    """插件 manifest 相关错误"""
    pass


class ManifestLoader:
    """
    插件 Manifest 加载器

    负责：
    - 读取 _manifest.json 文件
    - 进行基础字段校验
    - 将原始 manifest 数据解析为 PluginInfo（系统内部声明态对象）

    """

    def load_from_file(self, manifest_path: str | Path) -> PluginInfo:
        """
        从 _manifest.json 文件加载插件声明信息

        Args:
            manifest_path: manifest 文件路径

        Returns:
            PluginInfo: 解析后的插件声明对象
        """
        path = Path(manifest_path)

        if not path.exists():
            raise ManifestError(f"未找到 manifest 文件: {path}")

        if path.suffix != ".json":
            raise ManifestError(f"不支持的 manifest 格式: {path.suffix}")

        logger.debug(f"正在加载插件 manifest: {path}")

        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, dict):
            raise ManifestError(f"manifest 顶层结构必须是对象（文件: {path}）")

        return self._parse_manifest(raw_data, path)

    # ==========================
    # 内部解析逻辑
    # ==========================

    def _parse_manifest(self, raw: Dict[str, Any], path: Path) -> PluginInfo:
        """
        将原始 manifest 字典解析为 PluginInfo

        Args:
            raw: json 解析得到的原始字典
            path: manifest 文件路径（用于错误提示）

        Returns:
            PluginInfo
        """
        try:
            name = raw["name"]
            description = raw["description"]
        except KeyError as e:
            raise ManifestError(f"manifest 缺少必填字段 {e}（文件: {path}）")

        # 构建插件基础信息（声明态）
        info = PluginInfo(
            display_name=raw.get("display_name", name),
            name=name,
            description=description,
            version=raw.get("version", "1.0.0"),
            author=raw.get("author", ""),
            enabled=raw.get("enabled", True),
            is_built_in=raw.get("is_built_in", False),
            metadata=raw.get("metadata", {}),
            license=raw.get("license", ""),
            homepage_url=raw.get("homepage_url", ""),
            repository_url=raw.get("repository_url", ""),
            keywords=raw.get("keywords", []),
            categories=raw.get("categories", []),
            min_host_version=raw.get("min_host_version", ""),
            max_host_version=raw.get("max_host_version", ""),
            config_file_name=raw.get("config_file_name", "config.toml"),
        )

        # --------------------------
        # 插件依赖（其他插件）
        # --------------------------

        info.dependencies = raw.get("dependencies", [])

        # --------------------------
        # Python 包依赖声明
        # --------------------------

        for dep in raw.get("python_dependencies", []):
            info.python_dependencies.append(
                PythonDependency(
                    package_name=dep["package"],
                    version=dep.get("version", ""),
                    optional=dep.get("optional", False),
                    description=dep.get("description", ""),
                    install_name=dep.get("install_name", ""),
                )
            )

        # --------------------------
        # 工具声明
        # --------------------------

        self._parse_tools(raw, info)

        return info

    # ==========================
    # 工具解析
    # ==========================

    def _parse_tools(self, raw: Dict[str, Any], info: PluginInfo) -> None:
        """
        解析插件声明的工具信息（不加载实现）

        manifest 示例：
        tools = [{"name": "get_weather", "description": "...", "parameters": [...]}]

        Args:
            raw: 原始 manifest 数据
            info: PluginInfo 对象（就地填充）
        """
        tools = raw.get("tools", [])

        if not isinstance(tools, list):
            raise ManifestError("tools 字段必须是数组")

        for tool in tools:
            if not isinstance(tool, dict):
                raise ManifestError("tools 中的每一项都必须是对象")

            name = tool.get("name")
            description = tool.get("description", "")
            parameters_raw = tool.get("parameters", [])

            if not isinstance(name, str) or not name.strip():
                raise ManifestError("tools[].name 必须是非空字符串")
            if not isinstance(description, str):
                raise ManifestError(f"tools[{name}].description 必须是字符串")
            if not isinstance(parameters_raw, list):
                raise ManifestError(f"tools[{name}].parameters 必须是数组")

            parameters: list[ToolParameter] = []
            for param in parameters_raw:
                if not isinstance(param, dict):
                    raise ManifestError(f"tools[{name}].parameters 中每一项都必须是对象")

                param_name = param.get("name")
                param_type = param.get("type")
                param_description = param.get("description", "")
                param_required = bool(param.get("required", False))
                param_choices = param.get("choices")

                if not isinstance(param_name, str) or not param_name.strip():
                    raise ManifestError(f"tools[{name}].parameters[].name 必须是非空字符串")
                if not isinstance(param_type, str) or not param_type.strip():
                    raise ManifestError(f"tools[{name}].parameters[{param_name}].type 必须是非空字符串")
                if not isinstance(param_description, str):
                    raise ManifestError(f"tools[{name}].parameters[{param_name}].description 必须是字符串")
                if param_choices is not None:
                    if not isinstance(param_choices, list) or not all(isinstance(x, str) for x in param_choices):
                        raise ManifestError(
                            f"tools[{name}].parameters[{param_name}].choices 必须是字符串数组或 null"
                        )

                parameters.append(
                    ToolParameter(
                        name=param_name,
                        type=param_type,
                        description=param_description,
                        required=param_required,
                        choices=param_choices,
                    )
                )

            info.tools.append(
                ToolInfo(
                    name=name,
                    description=description,
                    tool_parameters=parameters,
                )
            )

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
                # 检查tools数组
                if "tools" in plugin_info:
                    tools = plugin_info["tools"]
                    if not isinstance(tools, list):
                        self.validation_errors.append("plugin_info.tools应为数组格式")
                    else:
                        for i, tool in enumerate(tools):
                            if not isinstance(tool, dict):
                                self.validation_errors.append(f"plugin_info.tools[{i}]应为对象")
                            else:
                                # 检查工具必需字段（不再要求 type）
                                for tool_field in ["name", "description"]:
                                    if tool_field not in tool or not tool[tool_field]:
                                        self.validation_errors.append(
                                            f"plugin_info.tools[{i}]缺少必需字段: {tool_field}"
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
