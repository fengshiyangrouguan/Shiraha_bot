# config/config_loader.py

import toml
from pathlib import Path
from typing import Dict, Any, Union

# --- 核心：SimpleNamespace 替代方案 ---
class ConfigNamespace:
    """
    一个简单的命名空间类，允许通过点号访问字典键。
    解决了你在工厂函数中不得不使用 .__dict__.items() 的痛点。
    现在你可以使用 config.platform.qq.enabled 了。
    """
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                # 递归处理子字典
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)
    
    # 消除工厂函数中依赖 __dict__ 的特殊情况：提供一个干净的迭代接口
    def items(self):
        """返回不包含内部属性的键值对。"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    # 支持打印
    def __repr__(self):
        return f"ConfigNamespace({self.__dict__})"


def load_all_config(config_path: Path) -> ConfigNamespace:
    """
    加载 TOML 配置文件并转换为 ConfigNamespace 对象。
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置路径不存在: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config: Dict[str, Any] = toml.load(f)
            
        # 转换为我们简洁的命名空间对象
        return ConfigNamespace(raw_config)

    except toml.TomlDecodeError as e:
        # 实用主义：清晰地报告配置解析错误
        raise ValueError(f"TOML 配置解析失败，请检查语法: {config_path}. 错误: {e}")
    except Exception as e:
        raise RuntimeError(f"加载配置时发生意外错误: {e}")

# --- 外部依赖检查：实用主义 ---
# 别忘了在你的环境中安装： pip install toml