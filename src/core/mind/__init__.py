"""
Mind 模块 - 主脑提示词拼接系统

提供统一的提示词拼接服务，用于构建完整的主脑上下文
"""
from .mind_core import Mind, get_mind, clear_mind

__all__ = ["Mind", "get_mind", "clear_mind"]
