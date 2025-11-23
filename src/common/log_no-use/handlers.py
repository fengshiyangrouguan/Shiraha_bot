# log/handlers.py - 自定义日志处理器

from PySide6.QtCore import QObject, Signal
import logging

# 1. 新增一个只负责信号发射的 QObject 类
class LogSignalEmitter(QObject):
    new_log = Signal(str)


class QtLogHandler(logging.Handler):
    """
    Qt 日志处理器：通过 Signal 将日志发送到 GUI 组件
    可被多个控件连接 new_log 信号，实现日志实时显示
    """
    new_log = Signal(str)

    def __init__(self):
        super().__init__()
        self.emitter = LogSignalEmitter() 

    @property
    def new_log(self):
        return self.emitter.new_log
    
    def emit(self, record):
        """将日志记录格式化后通过信号发出"""
        try:
            msg = self.format(record)
            self.new_log.emit(msg)
        except Exception:
            self.handleError(record)