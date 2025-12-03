import re
from typing import Dict, Tuple
from src.common.logger import get_logger
from src.core.version import CORE_VERSION
# if TYPE_CHECKING:
#     from src.plugin_system.base.base_plugin import BasePlugin

logger = get_logger("version_comparator")


class VersionComparator:
    """版本号比较器

    支持语义化版本号比较，自动处理snapshot版本，并支持向前兼容性检查
    """

    # 版本兼容性映射表（硬编码）
    # 格式: {插件最大支持版本: [实际兼容的版本列表]}
    COMPATIBILITY_MAP = {
        "0.0.0": ["0.0.1"]
        # 可以根据需要添加更多兼容映射
        # "0.9.0": ["0.9.1", "0.9.2", "0.9.3"],  # 示例：0.9.x系列兼容
    }

    @staticmethod
    def normalize_version(version: str) -> str:
        """标准化版本号，移除snapshot标识

        Args:
            version: 原始版本号，如 "0.8.0-snapshot.1"

        Returns:
            str: 标准化后的版本号，如 "0.8.0"
        """
        if not version:
            return "0.0.0"

        # 移除snapshot部分
        normalized = re.sub(r"-snapshot\.\d+", "", version.strip())

        # 确保版本号格式正确
        if not re.match(r"^\d+(\.\d+){0,2}$", normalized):
            # 如果不是有效的版本号格式，返回默认版本
            return "0.0.0"

        # 尝试补全版本号
        parts = normalized.split(".")
        while len(parts) < 3:
            parts.append("0")
        normalized = ".".join(parts[:3])

        return normalized

    @staticmethod
    def parse_version(version: str) -> Tuple[int, int, int]:
        """解析版本号为元组

        Args:
            version: 版本号字符串

        Returns:
            Tuple[int, int, int]: (major, minor, patch)
        """
        normalized = VersionComparator.normalize_version(version)
        try:
            parts = normalized.split(".")
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            logger.warning(f"无法解析版本号: {version}，使用默认版本 0.0.0")
            return (0, 0, 0)

    @staticmethod
    def compare_versions(version1: str, version2: str) -> int:
        """比较两个版本号

        Args:
            version1: 第一个版本号
            version2: 第二个版本号

        Returns:
            int: -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        v1_tuple = VersionComparator.parse_version(version1)
        v2_tuple = VersionComparator.parse_version(version2)

        if v1_tuple < v2_tuple:
            return -1
        elif v1_tuple > v2_tuple:
            return 1
        else:
            return 0

    @staticmethod
    def check_forward_compatibility(current_version: str, max_version: str) -> Tuple[bool, str]:
        """检查向前兼容性（仅使用兼容性映射表）

        Args:
            current_version: 当前版本
            max_version: 插件声明的最大支持版本

        Returns:
            Tuple[bool, str]: (是否兼容, 兼容信息)
        """
        current_normalized = VersionComparator.normalize_version(current_version)
        max_normalized = VersionComparator.normalize_version(max_version)

        # 检查兼容性映射表
        if max_normalized in VersionComparator.COMPATIBILITY_MAP:
            compatible_versions = VersionComparator.COMPATIBILITY_MAP[max_normalized]
            if current_normalized in compatible_versions:
                return True, f"根据兼容性映射表，版本 {current_normalized} 与 {max_normalized} 兼容"

        return False, ""

    @staticmethod
    def is_version_in_range(version: str, min_version: str = "", max_version: str = "") -> Tuple[bool, str]:
        """检查版本是否在指定范围内，支持兼容性检查

        Args:
            version: 要检查的版本号
            min_version: 最小版本号（可选）
            max_version: 最大版本号（可选）

        Returns:
            Tuple[bool, str]: (是否兼容, 错误信息或兼容信息)
        """
        if not min_version and not max_version:
            return True, ""

        version_normalized = VersionComparator.normalize_version(version)

        # 检查最小版本
        if min_version:
            min_normalized = VersionComparator.normalize_version(min_version)
            if VersionComparator.compare_versions(version_normalized, min_normalized) < 0:
                return False, f"版本 {version_normalized} 低于最小要求版本 {min_normalized}"

        # 检查最大版本
        if max_version:
            max_normalized = VersionComparator.normalize_version(max_version)
            comparison = VersionComparator.compare_versions(version_normalized, max_normalized)

            if comparison > 0:
                # 严格版本检查失败，尝试兼容性检查
                is_compatible, compat_msg = VersionComparator.check_forward_compatibility(
                    version_normalized, max_normalized
                )

                if not is_compatible:
                    return False, f"版本 {version_normalized} 高于最大支持版本 {max_normalized}，且无兼容性映射"

                logger.info(f"版本兼容性检查：{compat_msg}")
                return True, compat_msg
        return True, ""

    @staticmethod
    def get_current_host_version() -> str:
        """获取当前主机应用版本

        Returns:
            #str: 当前版本号
        """
        return VersionComparator.normalize_version(CORE_VERSION)

    @staticmethod
    def add_compatibility_mapping(base_version: str, compatible_versions: list) -> None:
        """动态添加兼容性映射

        Args:
            base_version: 基础版本（插件声明的最大支持版本）
            compatible_versions: 兼容的版本列表
        """
        base_normalized = VersionComparator.normalize_version(base_version)
        VersionComparator.COMPATIBILITY_MAP[base_normalized] = [
            VersionComparator.normalize_version(v) for v in compatible_versions
        ]
        logger.info(f"添加兼容性映射：{base_normalized} -> {compatible_versions}")

    @staticmethod
    def get_compatibility_info() -> Dict[str, list]:
        """获取当前的兼容性映射表

        Returns:
            Dict[str, list]: 兼容性映射表的副本
        """
        return VersionComparator.COMPATIBILITY_MAP.copy()

