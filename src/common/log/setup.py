# log/setup.py - 日志系统初始化
from pathlib import Path
import logging
from datetime import datetime
from .config import LOG_DIR
from .formatters import PlainFormatter, EmojiFormatter

def setup_logging(logger_name="ROOT", gui_widget=None):
    """
    初始化指定名称的日志记录器
    :param logger_name: 日志记录器名称
    :param gui_widget: 主窗口（需有 qt_handler 属性，用于 GUI 日志显示）
    :return: 配置好的 logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # 清除旧 handlers

    # 1. GUI Handler：带 emoji
    if gui_widget and hasattr(gui_widget, 'qt_handler'):
        gui_handler = gui_widget.qt_handler
        gui_handler.setFormatter(EmojiFormatter(fmt="%(message)s"))
        gui_handler.setLevel(logging.INFO)
        logger.addHandler(gui_handler)
    return logger


def setup_global_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 全局日志文件
    global_log_file = log_file = log_dir / f"LOG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 创建 handler
    file_handler = logging.FileHandler(global_log_file, encoding='utf-8', mode='a')
    file_handler.setFormatter(PlainFormatter())
    
    # 添加到 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    return root_logger

