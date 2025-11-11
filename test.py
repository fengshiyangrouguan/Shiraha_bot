# /test.py
import asyncio
import sys
import os

# 将 src 目录添加到Python的模块搜索路径中
# 这样我们就可以从根目录的脚本中，轻松地导入src下的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.core.bot import MainSystem

async def main():
    """
    主异步函数，用于启动机器人并处理用户输入。
    """
    print("正在初始化 MainSystem...")
    main_system = MainSystem()
    print("Shiraha_bot 已准备就绪！(๑•̀ㅂ•́)و✧")
    print("输入 'exit' 或 'quit' 来退出程序。")
    print("-" * 30)

    # 为了测试方便，我们预设一个对话ID和用户ID
    # 在真实应用中，这些ID会由聊天平台提供
    stream_id = "test_group_001"
    user_id = "test_user_12345"

    while True:
        try:
            # 获取用户在控制台的输入
            message_content = input("你: ")

            if message_content.lower() in ["exit", "quit"]:
                print("正在关闭... (｡･ω･｡)ﾉ♡")
                break
            
            if not message_content:
                continue

            # 调用 MainSystem 的核心处理方法
            bot_reply = await main_system.handle_message(
                stream_id=stream_id,
                user_id=user_id,
                message_content=message_content
            )

            if bot_reply:
                print(f"小智: {bot_reply}")
            else:
                print("小智: ...（好像不知道该说什么）")

        except KeyboardInterrupt:
            print("\n正在关闭... (｡･ω･｡)ﾉ♡")
            break
        except Exception as e:
            print(f"\n发生了一个错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
