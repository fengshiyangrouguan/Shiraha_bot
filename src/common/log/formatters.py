# log/formatters.py - 自定义日志格式化器

import logging
from .config import get_emoji_for_msg

class PlainFormatter(logging.Formatter):
    """
    纯文本格式化器
    示例: 2025-09-20 19:30:01 - [ui] INFO - 日志系统已启动
    """
    def format(self, record):
        dt = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        module = record.name
        level = record.levelname
        msg = record.getMessage()
        return f"{dt} - [{module}] {level} - {msg}"


class EmojiFormatter(logging.Formatter):
    """
    GUI 日志格式化器：自动为消息添加 emoji（仅命中关键词时）
    """
    def format(self, record):
        msg = record.msg
        if isinstance(msg, str):
            emoji = get_emoji_for_msg(msg)
            if emoji and not msg.startswith(emoji):
                msg = f"{emoji} {msg}"

        # 临时替换消息，调用父类 format
        original_msg = record.msg
        record.msg = msg
        formatted = super().format(record)
        record.msg = original_msg
        return formatted