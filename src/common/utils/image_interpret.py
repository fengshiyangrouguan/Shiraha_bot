from datetime import datetime
import hashlib
import base64
import io
import random
import os
from typing import Optional, List, Tuple

from PIL import Image

import numpy as np
import json
from src.llm_api.factory import LLMRequestFactory
from src.system.di.container import container
from src.common.database.database_manager import DatabaseManager
from src.common.database.database_model import StickerDB
from src.common.event_model.sticker import Sticker
from src.cortices.qq_chat.chat.sticker_system.sticker_manager import StickerManager
from src.common.logger import get_logger
from src.common.config.config_service import ConfigService
from time import time
from sqlalchemy import select

logger = get_logger("ImageInterpreter")


        
async def interpret_image(base64_image_data: str) -> str:
    """
    解释图像内容，返回文本描述和情感标签列表。
    根据 is_sticker 标志分派到不同的处理函数。
    
    Args:
        base64_image_data: 图像的Base64编码字符串。
        is_sticker: 指示这是否是一个表情包。

    Returns:
        str: 图像的文本描述。
    """
    try:
        # 按需解析依赖
        llm_factory: LLMRequestFactory = container.resolve(LLMRequestFactory)
    except Exception:
        logger.warning("无法解析 ImageInterpreter 的依赖，可能不在正确的 contexts 中。")
        return "[图片]", []

    description: str = ""
    prefix: str = "发了一张图片："

    try:
        prompt = "请仅简单描述图片的视觉内容，不要添加任何额外的问候、感想或建议。"
        vlm_request = llm_factory.get_request("vlm")
        description, _ = await vlm_request.execute_with_image(
            prompt=prompt,
            base64_image_data=base64_image_data
        )
        
        return f"{prefix}{description}"

    except Exception as e:
        logger.error(f"解释图像时出错: {e}", exc_info=True)
        return "[图片]", []


async def interpret_sticker(base64_image_data: str) -> str:
    """
    解释图像内容，返回文本描述和情感标签列表。
    根据 is_sticker 标志分派到不同的处理函数。
    
    Args:
        base64_image_data: 图像的Base64编码字符串。
        is_sticker: 指示这是否是一个表情包。

    Returns:
        str: 图像的文本描述。
    """
    try:
        # 按需解析依赖
        db_manager: DatabaseManager = container.resolve(DatabaseManager)
        llm_factory: LLMRequestFactory = container.resolve(LLMRequestFactory)
        config_service: ConfigService = container.resolve(ConfigService)
        sticker_manager: StickerManager = container.resolve(StickerManager)
    except Exception:
        logger.warning("无法解析 ImageInterpreter 的依赖，可能不在正确的 contexts 中。")
        return "[表情包]"

    description: str = ""
    emotions: List[str] = []
    prefix: str = "发了一个表情包："

    try:
        image_bytes = base64.b64decode(base64_image_data)
        image_hash = hashlib.md5(image_bytes).hexdigest()
        image_format = Image.open(io.BytesIO(image_bytes)).format.lower()
        logger.debug(f"新表情包 (hash: {image_hash[:10]}..., format: {image_format}), 调用VLM进行分析。")
        sticker_search = sticker_manager.get_sticker_by_hash(image_hash)
        if sticker_search:
            logger.info(f"表情包 {image_hash[:10]} 已存在，直接使用缓存的描述和情感标签。")
            return f"{prefix}{sticker_search.description}"
        # --- VLM视觉分析 ---
        if image_format in ["gif", "GIF"]:
            base64_image_data = _transform_gif(base64_image_data)
            prompt = "这是一个动态图表情包，请简单描述表情包的视觉内容和所表达的情感，精简回答。请注意这是一个多帧的动态图，黑色背景代表透明。"
        else:
            prompt = "这是一个表情包，请简单描述表情包的视觉内容和所表达的情感，精简回答"
        
        vlm_request = llm_factory.get_request("vlm")
        description, _ = await vlm_request.execute_with_image(
            prompt=prompt,
            base64_image_data=base64_image_data
        )
        # --- LLM情感分析 ---
        emotion_prompt = f"""
这是一个聊天场景中的表情包描述：'{description}'

请你识别这个表情包的含义和适用场景，从互联网梗、meme 以及贴吧/微博/小红书的视角，提取该表情包传达的核心感觉或情绪氛围。
要求：
只输出感觉性词语，每个词语不超过 5 个字。
你可以关注其幽默和讽刺意味，要精准捕捉其幽默、讽刺或深层的情感共鸣。
请直接输出描述性词语，不要出现任何其他内容，如果有多个词用逗号分隔。
        """
        llm_emotion_request = llm_factory.get_request("replyer")
        emotions_text, _ = await llm_emotion_request.execute(
            prompt=emotion_prompt, temperature=0.3, max_tokens=512
        )

        emotions = [e.strip() for e in emotions_text.split(",") if e.strip()]
        if len(emotions) > 3:
            emotions = random.sample(emotions, 2)

        llm_embedding_request = llm_factory.get_request("embedding")
        embedding, _ = await llm_embedding_request.execute_embedding(description)
        logger.info(f"表情包分析结果: 描述: {description[:50]}... -> 情感标签: {emotions} -> embedding: {embedding}")

        # --- 存入数据库和缓存 ---
        current_timestamp = time()
        filename = f"{int(current_timestamp)}_{image_hash[:8]}.{image_format}"
        sticker_dir = "data\sticker"
        os.makedirs(sticker_dir, exist_ok=True)
        file_path = os.path.join(sticker_dir, filename)
        try:
            # 保存表情到本地并存储到数据库
            with open(file_path, "wb") as f:
                f.write(image_bytes)           
            await _create_and_cache_sticker(
                db_manager=db_manager, 
                image_hash=image_hash, 
                file_path=file_path,
                file_format=image_format,
                description=description, 
                emotion=emotions, 
                embedding=embedding,
                last_used_time=current_timestamp
            )
            new_sticker = Sticker(
                sticker_hash=image_hash,
                file_path=file_path,
                file_format=image_format,
                description=description,
                emotions=emotions,
                embedding=embedding,
                last_used_time=current_timestamp,
                usage_count=0)
            
            await sticker_manager.add_sticker_to_cache(new_sticker)
        except Exception as e:
            logger.error(f"保存表情包文件时出错: {e}", exc_info=True)

        
        return f"{prefix}{description}"

    except Exception as e:
        logger.error(f"解释表情包时出错: {e}", exc_info=True)
        return "[表情包]"


async def _create_and_cache_sticker(db_manager: DatabaseManager, image_hash: str, file_path: str, file_format: str, description: str, emotions: List[str], embedding: List[float],last_used_time: float):
    """
    (内部函数) 创建新的Sticker记录，并将其保存到数据库和内存缓存。
    """


    logger.info(f"正在为新表情包创建记录 (hash: {image_hash[:10]}...)")
    
    new_sticker_db = StickerDB(
        sticker_hash=image_hash,
        file_path=file_path,
        file_format=file_format,
        description=description,
        emotion=json.dumps(emotions, ensure_ascii=False), # 使用传入的情感标签
        embedding=embedding,
        is_registered=False,
        last_used_time=last_used_time,
        usage_count='0',
    )

    async with await db_manager.get_session() as session:
        # 1. 先查一下这个 hash 是否已存在
        #TODO:一个临时策略，真正的解决方法是根本不处理重复的表情包，需要sticker_manager支持
        stmt = select(StickerDB).where(StickerDB.sticker_hash == image_hash)
        result = await session.execute(stmt)
        existing_sticker = result.scalar_one_or_none()

        if existing_sticker:
            logger.info(f"表情包 {image_hash[:10]} 已存在，跳过插入。")
            return
        
        session.add(new_sticker_db)
        await session.commit()
    logger.info(f"新表情包记录已创建并缓存 (hash: {image_hash[:10]}...)")

def _transform_gif(gif_base64: str, similarity_threshold: float = 1000.0, max_frames: int = 15) -> Optional[str]:
    # sourcery skip: use-contextlib-suppress
    """将GIF转换为水平拼接的静态图像, 跳过相似的帧

    Args:
        gif_base64: GIF的base64编码字符串
        similarity_threshold: 判定帧相似的阈值 (MSE)，越小表示要求差异越大才算不同帧，默认1000.0
        max_frames: 最大抽取的帧数，默认15

    Returns:
        Optional[str]: 拼接后的JPG图像的base64编码字符串, 或者在失败时返回None
    """
    try:
        # 确保base64字符串只包含ASCII字符
        if isinstance(gif_base64, str):
            gif_base64 = gif_base64.encode("ascii", errors="ignore").decode("ascii")
        # 解码base64
        gif_data = base64.b64decode(gif_base64)
        gif = Image.open(io.BytesIO(gif_data))

        # 收集所有帧
        all_frames = []
        try:
            while True:
                gif.seek(len(all_frames))
                # 确保是RGB格式方便比较
                frame = gif.convert("RGB")
                all_frames.append(frame.copy())
        except EOFError:
            pass  # 读完啦

        if not all_frames:
            logger.warning("GIF中没有找到任何帧")
            return None  # 空的GIF直接返回None

        # --- 新的帧选择逻辑 ---
        selected_frames = []
        last_selected_frame_np = None

        for i, current_frame in enumerate(all_frames):
            current_frame_np = np.array(current_frame)

            # 第一帧总是要选的
            if i == 0:
                selected_frames.append(current_frame)
                last_selected_frame_np = current_frame_np
                continue

            # 计算和上一张选中帧的差异（均方误差 MSE）
            if last_selected_frame_np is not None:
                mse = np.mean((current_frame_np - last_selected_frame_np) ** 2)
                # logger.debug(f"帧 {i} 与上一选中帧的 MSE: {mse}") # 可以取消注释来看差异值

                # 如果差异够大，就选它！
                if mse > similarity_threshold:
                    selected_frames.append(current_frame)
                    last_selected_frame_np = current_frame_np
                    # 检查是不是选够了
                    if len(selected_frames) >= max_frames:
                        # logger.debug(f"已选够 {max_frames} 帧，停止选择。")
                        break
            # 如果差异不大就跳过这一帧啦

        # --- 帧选择逻辑结束 ---

        # 如果选择后连一帧都没有（比如GIF只有一帧且后续处理失败？）或者原始GIF就没帧，也返回None
        if not selected_frames:
            logger.warning("处理后没有选中任何帧")
            return None

        # logger.debug(f"总帧数: {len(all_frames)}, 选中帧数: {len(selected_frames)}")

        # 获取选中的第一帧的尺寸（假设所有帧尺寸一致）
        frame_width, frame_height = selected_frames[0].size

        # 计算目标尺寸，保持宽高比
        target_height = 200  # 固定高度
        # 防止除以零
        if frame_height == 0:
            logger.error("帧高度为0，无法计算缩放尺寸")
            return None
        target_width = int((target_height / frame_height) * frame_width)
        # 宽度也不能是0
        if target_width == 0:
            logger.warning(f"计算出的目标宽度为0 (原始尺寸 {frame_width}x{frame_height})，调整为1")
            target_width = 1

        # 调整所有选中帧的大小
        resized_frames = [
            frame.resize((target_width, target_height), Image.Resampling.LANCZOS) for frame in selected_frames
        ]

        # 创建拼接图像
        total_width = target_width * len(resized_frames)
        # 防止总宽度为0
        if total_width == 0 and resized_frames:
            logger.warning("计算出的总宽度为0，但有选中帧，可能目标宽度太小")
            # 至少给点宽度吧
            total_width = len(resized_frames)
        elif total_width == 0:
            logger.error("计算出的总宽度为0且无选中帧")
            return None

        combined_image = Image.new("RGB", (total_width, target_height))

        # 水平拼接图像
        for idx, frame in enumerate(resized_frames):
            combined_image.paste(frame, (idx * target_width, 0))

        # 转换为base64
        buffer = io.BytesIO()
        combined_image.save(buffer, format="JPEG", quality=85)  # 保存为JPEG
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except MemoryError:
        logger.error("GIF转换失败: 内存不足，可能是GIF太大或帧数太多")
        return None  # 内存不够啦
    except Exception as e:
        logger.error(f"GIF转换失败: {str(e)}", exc_info=True)  # 记录详细错误信息
        return None  # 其他错误也返回None