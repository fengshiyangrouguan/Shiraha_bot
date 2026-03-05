import io
import os
import uuid
import time
import base64
import hashlib
import asyncio
from typing import Optional, Tuple

import PIL.Image as Image
from src.common.logger import get_logger
from src.llm_api.factory import LLMRequestFactory
from src.system.di.container import container

logger = get_logger("ImageInterprete")

class ImageInterpreter:
    def __init__(self, vlm_instance, image_dir: str):
        """
        初始化图片解释器
        :param vlm_instance: 传入已实例化的 VLM 请求对象
        :param image_dir: 图片本地存储根目录
        """
        self.vlm = vlm_instance
        self.IMAGE_DIR = image_dir
        self.llm_factory = container.resolve(LLMRequestFactory)


    async def get_image_description(self, image_base64: str, is_sticker: bool = True) -> str:
        """
        获取图片或表情包的情感/内容描述
        :param image_base64: 图片 base64 字符串
        :param is_sticker: 是否为表情包(Sticker/Emoji)，True 则执行情感浓缩，False 则仅做视觉描述
        """
        try:
            # 1. 基础预处理
            if isinstance(image_base64, str):
                image_base64 = image_base64.encode("ascii", errors="ignore").decode("ascii")
            
            image_bytes = base64.b64decode(image_base64)
            image_hash = hashlib.md5(image_bytes).hexdigest()
            
            
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    image_format = img.format.lower() if img.format else "jpg"
            except Exception:
                image_format = "jpg"

            # 2. 缓存层：优先从数据库/管理器查询
            description = await self._check_cache(image_hash, is_sticker)
            if description:
                return f"[表情包：{description}]" if is_sticker else f"[图片：{description}]"

            # 3. 第一步：VLM 视觉分析
            detailed_description = await self._run_vlm_analysis(image_base64, image_format, is_sticker)
            if not detailed_description:
                return "[表情包]" if is_sticker else "[图片]"

            # 4. 第二步：LLM 情感浓缩   (仅针对表情包)
            if is_sticker:
                final_label = await self._run_emotion_extraction(detailed_description)
            else:
                final_label = detailed_description

            # 5. 异步持久化：保存文件、记录数据库 (不阻塞主流程)
            asyncio.create_task(self._persist_data(
                image_hash, image_bytes, image_format, detailed_description, final_label, is_sticker
            ))

            return f"[表情包：{final_label}]" if is_sticker else f"[图片：{final_label}]"

        except Exception as e:
            logger.error(f"[EmojiInterpreter] 处理失败: {e}", exc_info=True)
            return "[表情包]" if is_sticker else "[图片]"

    async def _check_cache(self, image_hash: str, is_sticker: bool) -> Optional[str]:
        """多级缓存查询"""
        # A. 尝试 EmojiManager 注册表
        if is_sticker:
            try:
                from src.chat.emoji_system.emoji_manager import get_emoji_manager
                mgr = get_emoji_manager()
                tags = await mgr.get_emoji_tag_by_hash(image_hash)
                if tags:
                    return "，".join(tags)
            except Exception:
                pass

        # B. 尝试 ImageDescriptions 缓存表
        # 注意：这里需要你实现具体的数据库查询方法 _get_description_from_db
        # cached = self._get_description_from_db(image_hash, "emoji" if is_sticker else "image")
        # return cached
        return None

    async def _run_vlm_analysis(self, image_base64: str, img_format: str, is_sticker: bool) -> Optional[str]:
        """调用 VLM 获取详细描述"""
        if is_sticker:
            prompt = "这是一个表情包，请详细描述其视觉内容、文字以及隐含的互联网梗/meme情感，精简回答。"
            # 如果是 GIF，可以在这里调用你的 transform_gif 逻辑
            if img_format == "gif" and hasattr(self, 'transform_gif'):
                image_base64 = self.transform_gif(image_base64) or image_base64
        else:
            prompt = "请仅简单描述图片的视觉内容，不要添加任何额外的问候、感想或建议。"

        # 兼容你原流程中的 execute_with_image 接口
        description, _ = await self.vlm.execute_with_image(
            prompt=prompt,
            base64_image_data=image_base64
        )
        return description

    async def _run_emotion_extraction(self, detailed_description: str) -> str:
        """调用 LLM 将长描述浓缩为 1-2 个情感标签"""
        emotion_prompt = f"""
        基于表情包描述，提取1-2个最核心的情感/梗标签（如：委屈、无语、狂喜）。
        描述：'{detailed_description}'
        要求：直接输出标签，多个用逗号隔开，不要任何解释。
        """
        try:
            emotion_llm = llm_factory.get_request("vlm")
            result, _ = await emotion_llm.generate_response_async(emotion_prompt, temperature=0.3)
            if result:
                # 清洗可能的标点符号
                labels = [l.strip() for l in result.replace("，", ",").split(",") if l.strip()]
                return "，".join(labels[:2])
        except Exception as e:
            logger.warning(f"情感浓缩失败: {e}")
        
        return detailed_description[:10] # 降级方案

    async def _persist_data(self, img_hash, img_bytes, img_format, detailed_desc, final_label, is_sticker):
        """后台持久化：保存文件和数据库记录"""
        try:
            # 1. 保存本地文件
            sub_dir = "emoji" if is_sticker else "received"
            save_dir = os.path.join(self.IMAGE_DIR, sub_dir)
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f"{int(time.time())}_{img_hash[:8]}.{img_format}"
            file_path = os.path.join(save_dir, filename)
            
            with open(file_path, "wb") as f:
                f.write(img_bytes)

            # 2. 写入数据库 (此处需对接你的 ORM)
            # self._save_description_to_db(img_hash, final_label, "emoji" if is_sticker else "image")
            # logger.debug(f"已持久化图片数据: {img_hash}")

        except Exception as e:
            logger.error(f"持久化失败: {e}")

    def transform_gif(self, image_base64: str) -> Optional[str]:
        """处理 GIF 的占位逻辑，如果需要可以在此实现帧提取"""
        # TODO: 实现具体的 GIF 处理逻辑
        return image_base64