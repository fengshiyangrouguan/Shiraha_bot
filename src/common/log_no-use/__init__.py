# log/__init__.py - 日志系统公共 API

from .config import LOG_DIR
from .handlers import QtLogHandler 
from .setup import setup_logging, setup_global_logging

__all__ = [
    'setup_logging',
    'setup_global_logging',
    'QtLogHandler',
    'LOG_DIR',
]