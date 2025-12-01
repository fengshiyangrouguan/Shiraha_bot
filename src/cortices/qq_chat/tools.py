# a new file: src/cortices/qq_chat/tools.py
from src.cortices.manager import CortexManager

cortex_manager = CortexManager()

# 第二层: 主规划器的 "快捷工具"
@cortex_manager.tool(scope="main")
async def send_quick_qq_reply(chat_id: str, content: str) -> str:
    """
    向指定的QQ聊天对象（用户或群组）发送一条简单的、一次性的消息。
    适用于不需要深入对话的场景。
    """
    print(f"快捷工具: 正在向 {chat_id} 发送消息: {content}")
    # 在这里将会有调用QQ适配器发送消息的真实逻辑
    return f"消息 '{content}' 已成功发送至 {chat_id}。"

# 第二层: 主规划器的 "入口工具"
@cortex_manager.tool(scope="main")
async def enter_qq_chat_mode(chat_id: str) -> str:
    """
    当需要与某个QQ聊天对象进行复杂、多轮的对话时，进入此专属聊天模式。
    """
    print(f"入口工具: 准备进入与 {chat_id} 的专属聊天模式...")
    # 这里的实现将是创建并运行一个 QQChatSubPlanner
    # 目前，我们先用一个占位符代替
    return f"已进入与 {chat_id} 的专属聊天模式，后续操作由子规划器接管。（占位实现）"

# 第三层: 子规划器的 "精细操作工具"
@cortex_manager.tool(scope="qq_chat")
async def send_qq_emoji(emoji_id: str) -> str:
    """
    在当前的专属聊天中，发送一个指定的QQ表情。
    """
    print(f"精细工具: 正在发送表情: {emoji_id}")
    return f"表情 {emoji_id} 已成功发送。"
