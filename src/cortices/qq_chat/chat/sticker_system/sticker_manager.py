from typing import Dict, Optional, Set
from sqlalchemy.future import select
from src.common.database.database_manager import DatabaseManager
from src.common.database.database_model import StickerDB
from src.common.event_model.sticker import Sticker
from src.common.logger import get_logger

logger = get_logger("StickerManager")

class StickerManager:
    """
    负责在启动时从数据库加载所有表情包元数据到内存中，
    作为一个只读缓存，并提供快速检索服务。
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        初始化 StickerManager。
        Args:
            db_manager: 数据库管理器实例。
        """
        self.db_manager = db_manager
        self._stickers_by_hash: Dict[str, Sticker] = {}
        self._is_initialized = False

    async def start(self):
        """
        启动并初始化管理器，从数据库加载所有表情包对象。
        """
        if self._is_initialized:
            logger.info("StickerManager 已初始化，跳过。")
            return
        
        logger.info("StickerManager 正在启动...")
        await self._load_from_db()
        self._is_initialized = True
        logger.info(f"StickerManager 启动完成，加载了 {len(self._stickers_by_hash)} 个表情包。")

    async def _load_from_db(self):
        """
        从数据库加载所有表情包记录，并将它们转换为 Sticker 对象存储在内存中。
        """
        async with self.db_manager.get_session() as session:
            stmt = select(StickerDB)
            result = await session.execute(stmt)
            sticker_records = result.scalars().all()

            for record in sticker_records:
                if not record.image_hash or not record.id:
                    continue
                
                # 适配 StickerDB 的 emotion 字段 (string) 到 Sticker 的 keywords (List[str])
                keywords = record.emotion.split(',') if record.emotion else []

                sticker = Sticker(
                    sticker_id=record.id,
                    sticker_hash=record.image_hash,
                    file_format=record.format or "unknown",
                    keywords=keywords,
                )
                self._stickers_by_hash[sticker.sticker_hash] = sticker

    def get_sticker_by_hash(self, sticker_hash: str) -> Optional[Sticker]:
        """
        通过文件哈希值从内存缓存中快速检索 Sticker 对象。

        Args:
            file_hash: 文件的哈希值。

        Returns:
            如果找到，返回 Sticker 对象，否则返回 None。
        """
        return self._stickers_by_hash.get(sticker_hash)
    
    def add_sticker_to_cache(self, sticker: Sticker):
        """
        当外部系统创建一个新表情包时，将新的 Sticker 对象添加到内存缓存中。
        这避免了每次添加新表情包时都需要重启或重新扫描数据库。

        Args:
            sticker: 新创建的 Sticker 对象。
        """
        if sticker.file_hash in self._stickers_by_hash:
            logger.warning(f"尝试向缓存中添加已存在的 Sticker (hash: {sticker.file_hash[:10]}...)")
            return
        
        self._stickers_by_hash[sticker.file_hash] = sticker
        logger.debug(f"新表情包已添加至内存缓存: {sticker.file_hash[:10]}...")

