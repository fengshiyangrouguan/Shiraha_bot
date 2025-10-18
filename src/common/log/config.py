# log/config.py - 日志系统配置

from pathlib import Path

# 日志目录（基于当前工作目录）
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Emoji 映射表：关键词 → emoji
KEYWORD_TO_EMOJI = {
    ('已', '成功', '完成', '就绪', '准备'): '✅',
    ('失败', '错误', '异常'): '🔴',
    ('加载', '开始', '运行', '启动'): '🟡',
}

def get_emoji_for_msg(msg: str) -> str:
    """
    根据消息内容返回对应 emoji，没有匹配则返回空字符串
    :param msg: 日志消息
    :return: 匹配的 emoji，否则返回 ""
    """
    for keywords, emoji in KEYWORD_TO_EMOJI.items():
        if any(k in msg for k in keywords):
            return emoji
    return ""