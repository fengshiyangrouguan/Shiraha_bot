from typing import Dict, Any, List
from dataclasses import dataclass, field
from .component_info import ComponentInfo
@dataclass
class PythonDependency:
    """Python包依赖信息"""

    package_name: str  # 包名称
    version: str = ""  # 版本要求，例如: ">=1.0.0", "==2.1.3", ""表示任意版本
    optional: bool = False  # 是否为可选依赖
    description: str = ""  # 依赖描述
    install_name: str = ""  # 安装时的包名（如果与import名不同）

    def __post_init__(self):
        if not self.install_name:
            self.install_name = self.package_name

    def get_pip_requirement(self) -> str:
        """获取pip安装格式的依赖字符串"""
        if self.version:
            return f"{self.install_name}{self.version}"
        return self.install_name



@dataclass
class PluginInfo:
    """插件信息"""

    display_name: str  # 插件显示名称
    name: str  # 插件名称
    description: str  # 插件描述
    version: str = "1.0.0"  # 插件版本
    author: str = ""  # 插件作者
    enabled: bool = True  # 是否启用
    is_built_in: bool = False  # 是否为内置插件
    components: List[ComponentInfo] = field(default_factory=list)  # 包含的组件列表
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他插件
    python_dependencies: List[PythonDependency] = field(default_factory=list)  # Python包依赖
    config_file: str = ""  # 配置文件路径
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    # 新增：manifest相关信息
    manifest_data: Dict[str, Any] = field(default_factory=dict)  # manifest文件数据
    license: str = ""  # 插件许可证
    homepage_url: str = ""  # 插件主页
    repository_url: str = ""  # 插件仓库地址
    keywords: List[str] = field(default_factory=list)  # 插件关键词
    categories: List[str] = field(default_factory=list)  # 插件分类
    min_host_version: str = ""  # 最低主机版本要求
    max_host_version: str = ""  # 最高主机版本要求

    def __post_init__(self):
        if self.components is None:
            self.components = []
        if self.dependencies is None:
            self.dependencies = []
        if self.python_dependencies is None:
            self.python_dependencies = []
        if self.metadata is None:
            self.metadata = {}
        if self.manifest_data is None:
            self.manifest_data = {}
        if self.keywords is None:
            self.keywords = []
        if self.categories is None:
            self.categories = []

    def get_missing_packages(self) -> List[PythonDependency]:
        """检查缺失的Python包"""
        missing = []
        for dep in self.python_dependencies:
            try:
                __import__(dep.package_name)
            except ImportError:
                if not dep.optional:
                    missing.append(dep)
        return missing

    def get_pip_requirements(self) -> List[str]:
        """获取所有pip安装格式的依赖"""
        return [dep.get_pip_requirement() for dep in self.python_dependencies]

