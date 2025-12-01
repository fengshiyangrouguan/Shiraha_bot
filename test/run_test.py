import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import logging

# 将项目根目录添加到 sys.path，以确保可以正确导入 src 包
# 这个脚本期望被放置在 Shiraha_bot 目录下
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.platform.platform_manager import PlatformManager
from src.common.event_model.event import Event

# --- 基本日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 测试配置 ---

# 用于存储接收到的事件的队列
event_queue = asyncio.Queue()

# 1. 模拟的 post_method
async def mock_post_method(event: Event):
    """一个模拟的 post 方法，它将事件放入队列中以便测试验证。"""
    logger.info(f"事件已捕获: {event.event_type} from {event.platform}")
    await event_queue.put(event)

# 2. 模拟的平台配置
TEST_PORT = 8081
mock_platform_config = [
    {
        "name": "qq_napcat",
        "enabled": True,
        "id": "test_qq_instance",
        "host": "127.0.0.1",
        "port": TEST_PORT
    }
]

# --- 主运行函数 (已修改为永久循环) ---

async def main():
    """
    程序主入口，初始化平台管理器并进入永久事件监听循环。
    """
    logger.info("初始化平台管理器...")
    manager = PlatformManager(post_method=mock_post_method, platform_configs=mock_platform_config)
    
    try:
        # 启动适配器
        logger.info("加载并启动适配器...")
        manager.load_adapters()
        manager.start_all()
        await asyncio.sleep(0.5) # 给予服务器充足时间启动

        # 提示用户操作
        logger.info("="*50)
        logger.info(f"✅ QQ 适配器已启动，正在 ws://127.0.0.1:{TEST_PORT} 上持续监听...")
        logger.info("请发送消息进行测试。使用 Ctrl+C 来停止脚本。")
        logger.info("="*50)

        # 核心修改：将单次等待改为永久循环
        while True:
            received_event: Event = await event_queue.get() 
            
            logger.info("\n--- 收到新事件并开始验证 ---")

            # --- 开始验证和处理（逻辑保持不变） ---
            try:
                assert received_event is not None, "接收到的事件不应为 None"
                assert received_event.event_type == "message", "事件类型应为 'message'"
                #assert received_event.platform == "test_qq_instance", "事件来源 ID 不匹配"
                data = None
                if received_event.event_type == "message":
                    data = received_event.event_data.raw_message
                assert data is not None, "事件数据不应为 None"

                
                # logger.info(f"完整事件数据: {json.dumps(data, ensure_ascii=False,indent=2)}")
                logger.info(f"解析后的event数据: {received_event}")
                
                logger.info("✅ 事件处理和验证流程通过。")
                
            except AssertionError as e:
                logger.error(f"❌ 事件验证未通过: {e}")
            except Exception as e:
                logger.error(f"❌ 处理事件时发生意外错误: {e}")
            
            # 标记任务完成，并循环回去等待下一个事件
            event_queue.task_done()
            logger.info("--- 等待下一个事件 ---\n")

    except asyncio.CancelledError:
        # 当 asyncio.run() 收到 KeyboardInterrupt 时，会发送此信号
        logger.info("主循环被取消 (Ctrl+C 信号)。")
    finally:
        # 清理工作
        logger.info("正在优雅地停止所有适配器...")
        # 确保 manager 在 try 块中被初始化
        if 'manager' in locals():
             await manager.stop_all()
        logger.info("程序执行完毕。")


if __name__ == "__main__":
    try:
        # asyncio.run() 会处理 KeyboardInterrupt，并在内部触发 CancelledError
        asyncio.run(main())
    except KeyboardInterrupt:
        # 外部捕获以确保程序能干净退出，虽然内部的 finally 块会处理大部分清理
        logger.info("脚本被用户中断。")