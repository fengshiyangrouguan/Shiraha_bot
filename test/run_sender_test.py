import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

# 假设您的项目结构和导入路径设置是正确的
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.platform.platform_manager import PlatformManager
from src.platform.sources.qq_napcat.service.message_service import NapcatMessageService
from src.common.event_model.event import Event, ConversationInfo
from src.common.event_model.event_data import Message, MessageSegment
from src.common.event_model.info_data import UserInfo,ConversationInfo
from src.platform.sources.qq_napcat.utils.msg_api_build import (
    build_at_seg,
    build_face_seg,
    build_image_seg,
    build_reply_seg,
    build_text_seg,
    build_sticker_seg
)


# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 测试配置 ---
TEST_PLATFORM_ID = "test_qq_instance"
TEST_PORT = 8081
event_queue = asyncio.Queue()

mock_platform_config = [
    {
        "name": "qq_napcat",
        "enabled": True,
        "id": TEST_PLATFORM_ID,
        "host": "127.0.0.1",
        "port": TEST_PORT
    }
]

# 1. 模拟的 post_method (接收标准化 Event)
async def mock_post_method(event: Event):
    """适配器调用此方法推送标准化后的 Event 对象。"""
    logger.info(f"[POST] 事件已推送: {event.event_type} from {event.platform}")
    await event_queue.put(event)

# 2. 核心回声逻辑
async def echo_back_message(message_service:NapcatMessageService, event: Event):
    """
    根据接收到的 MessageEvent，构造包含相同内容的 segments 并发送回原处。
    """
    conversation_info = event.conversation_info
    user_info:UserInfo = event.user_info
    segments: List[Dict[str, Any]] = []
    message:Message = event.event_data
    # 消息段列表，假设位于 event_data.message
    message_segments:MessageSegment = message.segments

    logger.info(f"--- 解析事件 ID:{event.event_id} ---")

    # --- B. 内容回声 (Text, @, Image) ---
    for seg in message_segments:
        seg_type = seg.type
        seg_data = seg.data

        if seg_type == "reply":
            target_message_id = event.event_data.message_id
            if target_message_id:
                logger.info(f"-> 发现 reply 消息段，构造 reply 消息段: {target_message_id}")
                segments.append(build_reply_seg(target_message_id))  

        elif seg_type == "at":
            target_id = event.user_info.user_id
            if target_id:
                logger.info(f"-> 发现 @ 消息段，构造 @ 消息段: {target_id}")
                segments.append(build_at_seg(target_id))   

        elif seg_type == "text":
            text_content = seg_data
            logger.info(f"-> 发现文本段，构造 text 消息段: {text_content[:20]}...")
            # 增加前缀以示区分
            segments.append(build_text_seg(text_content))
                

                
        elif seg_type == "image":
            # 假设 image_seg 支持直接使用 file 字段 (file_id 或 url)
            image_base64 = seg_data
            if image_base64:
                logger.info(f"-> 发现图片，构造 image 消息段")
                segments.append(build_image_seg(image_base64))
            else:
                logger.warning("发送图片失败")
        elif seg_type == "sticker":
            image_base64 = seg_data
            if image_base64:
                logger.info(f"-> 发现表情包，构造 sticker 消息段")
                segments.append(build_sticker_seg(image_base64))
            else:
                logger.warning("发送表情包失败")
            

        elif seg_type == "face":
            face_id = seg_data
            segments.append(build_face_seg(face_id))


    if not segments:
        logger.warning(f"无法构造回声消息段，可能是空消息或不支持的类型。类型：{seg_type}")
        return

    # --- C. 调用 message_manager 发送 ---
    try:
        conv_type = conversation_info.conversation_type
        conv_id = conversation_info.conversation_id
        logger.info(f"-> 准备发送回声消息到 {conv_type}:{conv_id}, 包含 {len(segments)} 段")
        
        # 假设 message_manager 的核心发送方法是 send(conversation_info, segments)
        await message_service.send_message(conversation_info, segments)
        logger.info("-> 回声消息发送成功。")
    except Exception as e:
        logger.error(f"❌ 回声消息发送失败: {e}", exc_info=True)


# --- 主运行函数 ---

async def main():
    logger.info("初始化平台管理器...")
    manager = PlatformManager(post_method=mock_post_method, platform_configs=mock_platform_config)
    
    qq_adapter = None
    message_api = None
    
    try:
        manager.load_adapters()
        manager.start_all()
        await asyncio.sleep(1) # 给予服务器充足时间启动

        # 获取适配器实例和 MessageManager
        qq_adapter = manager.get_adapter(TEST_PLATFORM_ID)

        # MessageManager 挂载在 adapter 的 message_api 属性上
        if qq_adapter and hasattr(qq_adapter, 'message_api'):
             message_api = qq_adapter.message_api 
        
        if not message_api:
            logger.error("❌ 无法获取 MessageManager 实例，回声功能将无法工作！")
            return

        logger.info("="*50)
        logger.info(f"✅ QQ 适配器已启动，正在 ws://127.0.0.1:{TEST_PORT} 上持续监听...")
        logger.info(f"🤖 机器人已就绪，收到消息将发送回声。")
        logger.info("="*50)

        # 核心事件监听循环
        while True:
            received_event: Event = await event_queue.get() 
            
            # 确认是消息事件且是 MessageEvent 类型
            if received_event.event_type == "message" and isinstance(received_event, Event):
                await echo_back_message(message_api, received_event)
            else:
                 logger.info(f"收到非消息事件: {received_event.event_type}，跳过回声。")
            
            event_queue.task_done()
            logger.info("--- 等待下一个事件 ---\n")

    except asyncio.CancelledError:
        logger.info("主循环被取消 (Ctrl+C 信号)。")
    except Exception as e:
        logger.error(f"主程序发生意外错误: {e}", exc_info=True)
    finally:
        logger.info("正在优雅地停止所有适配器...")
        if manager:
            await manager.stop_all()
        logger.info("程序执行完毕。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("脚本被用户中断。")