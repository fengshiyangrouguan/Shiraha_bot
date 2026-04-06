"""
Cortex Config Loader - 兼容性模块

已移动到 src.cortex_system.cortex_config_loader
保留此文件以保证向后兼容
"""
from src.cortex_system.cortex_config_loader import load_cortex_config, BaseCortexConfigSchema

__all__ = ["load_cortex_config", "BaseCortexConfigSchema"]
