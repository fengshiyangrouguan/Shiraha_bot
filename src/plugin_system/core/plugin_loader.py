from pathlib import Path
from typing import Dict

from src.plugin_system.base import PluginInfo
from src.plugin_system.utils import ManifestLoader, ManifestError
from src.common.logger import get_logger

logger = get_logger("plugin_system")

class PluginLoader:
    """
    插件 Loader（声明态）

    职责：
    - 扫描 plugins 目录
    - 查找 _manifest.json
    - 解析为 PluginInfo
    - 不 import / 不执行插件代码
    """

    def __init__(self, plugin_root: Path):
        self.plugin_root = plugin_root
        self.manifest_loader = ManifestLoader()

    def load_plugins(self) -> Dict[str, PluginInfo]:
        """
        扫描插件目录并加载所有插件声明信息

        Returns:
            Dict[str, PluginInfo]: plugin_name -> PluginInfo
        """
        logger.info(f"开始扫描插件目录: {self.plugin_root}")

        self.plugin_root.mkdir(exist_ok=True)

        plugin_infos: Dict[str, PluginInfo] = {}

        for folder in self.plugin_root.iterdir():
            if not folder.is_dir():
                continue
            
            # 检查必要的文件
            plugin_file = folder / "plugin.py"
            manifest_file = folder / "_manifest.json"
            
            if not plugin_file.exists():
                logger.debug(f"跳过没有 plugin.py 的目录: {folder.name}")
                continue
            
            if not manifest_file.exists():
                logger.debug(f"跳过没有 _manifest.json 的目录: {folder.name}")
                continue

            try:
                info = self._load_single_manifest(folder)
            except ManifestError as e:
                logger.error(f"插件 '{folder.name}' manifest 错误: {e}")
                continue
            except Exception as e:
                logger.error(f"加载插件 '{folder.name}' 失败: {e}")
                continue

            if not info.enabled:
                logger.info(f"插件 '{info.name}' 已禁用，跳过加载")
                continue

            if info.name in plugin_infos:
                logger.error(f"插件名冲突: {info.name}")
                continue

            plugin_infos[info.name] = info
            logger.info(f"发现插件声明: {info.name}")

        logger.info(f"插件扫描完成，共发现 {len(plugin_infos)} 个插件声明")
        return plugin_infos

    def _load_single_manifest(self, folder: Path) -> PluginInfo:
        """
        加载单个插件目录的 manifest

        Args:
            folder: 插件目录路径

        Returns:
            PluginInfo
        """
        manifest_path = folder / "_manifest.json"
        info = self.manifest_loader.load_from_file(manifest_path)

        # 记录插件目录
        info.metadata["plugin_dir"] = folder

        return info
        