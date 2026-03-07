import numpy as np
from typing import Dict, Optional, Set, List
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

        self._embedding_matrix: Optional[np.ndarray] = None
        self._all_hashes: Optional[list] = None

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
        async with await self.db_manager.get_session() as session:
            stmt = select(StickerDB)
            result = await session.execute(stmt)
            sticker_records = result.scalars().all()

            for record in sticker_records:
                if not record.sticker_hash:
                    continue
                
                # 适配 StickerDB 的 emotion 字段 (string) 到 Sticker 的 keywords (List[str])
                emotions = record.emotion.split(',') if record.emotion else []

                sticker = Sticker(
                    sticker_hash=record.sticker_hash,
                    file_path=record.file_path,
                    file_format=record.file_format or "unknown",
                    description=record.description or "",
                    emotion=emotions,
                    embedding=record.embedding or [],
                    last_used_time=record.last_used_time,
                    usage_count=record.usage_count or 0
                )
                self._stickers_by_hash[sticker.sticker_hash] = sticker
                logger.debug(f"已加载表情包: [{sticker.description[:10]}]")
            self._rebuild_search_index()

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
        if sticker.sticker_hash in self._stickers_by_hash:
            logger.warning(f"尝试向缓存中添加已存在的 Sticker (hash: {sticker.sticker_hash[:10]}...)")
            return
        
        self._stickers_by_hash[sticker.sticker_hash] = sticker

        # 2. 向量处理
        if not sticker.embedding or len(sticker.embedding) == 0:
            return

        # 归一化新向量
        new_vec = np.array(sticker.embedding, dtype=np.float32)
        new_vec = new_vec / (np.linalg.norm(new_vec) + 1e-9)

        # 3. 追加或创建矩阵
        if self._embedding_matrix is None:
            # 第一次启动：从无到有
            self._embedding_matrix = new_vec.reshape(1, -1)
            self._all_hashes = [sticker.sticker_hash]
            logger.info("第一个向量已存入，搜索索引已激活。")
        else:
            # 维度检查（防止混合了不同模型的向量导致崩溃）
            if new_vec.shape[0] != self._embedding_matrix.shape[1]:
                logger.error(f"维度不匹配！预期 {self._embedding_matrix.shape[1]}, 实际 {new_vec.shape[0]}")
                return
                
            # 动态追加
            self._embedding_matrix = np.vstack([self._embedding_matrix, new_vec])
            self._all_hashes.append(sticker.sticker_hash)
            logger.info(f"已追加新向量，当前索引规模: {len(self._all_hashes)}")
        logger.debug(f"新表情包已添加至内存缓存: {sticker.description[:20]}...)")

    def search_stickers(self, query_embedding: List[float], top_k: int = 5) -> List[Sticker]:
        """使用余弦相似度检索最匹配的表情包"""
        # 1. 安全检查
        if self._embedding_matrix is None or not query_embedding:
            logger.warning("搜索请求失败：索引未准备好或查询向量无效。")
            return []

        # 2. 处理查询向量：归一化
        q = np.array(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)

        # 3. 计算点积 (Dot Product)
        # 由于矩阵已归一化，点积结果即为余弦相似度得分 [0.0 ~ 1.0]
        similarities = np.dot(self._embedding_matrix, q)

        # 4. 获取得分最高的索引
        # argsort 返回从小到大的索引，[::-1] 翻转为从大到小
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = similarities[idx]
            # 设置一个阈值，如果相似度太低（比如低于 0.3），可能就不太相关了
            if score < 0.25: 
                logger.info(f"相似度 {score:.4f} 低于阈值，停止继续匹配。")
                continue
                
            s_hash = self._all_hashes[idx]
            sticker = self._stickers_by_hash.get(s_hash)
            if sticker:
                logger.info(f"匹配成功: [Score: {score:.4f}] Hash: {s_hash[:8]}")
                results.append(sticker)

        best_sticker:Sticker = results[0] if results else None
        if best_sticker:
            logger.info(f"最佳匹配: {best_sticker.description[:10]}... (path: {best_sticker.file_path})")
            return best_sticker.file_path
        return None
    

    def _rebuild_search_index(self):
        """将内存中的 embedding 转换为 numpy 矩阵，增加空值保护"""
        stickers = list(self._stickers_by_hash.values())
        # 过滤：必须有 embedding，且维度必须正确（假设是 1536 维）
        valid_stickers = [s for s in stickers if s.embedding and len(s.embedding) > 0]
        
        if not valid_stickers:
            logger.warning("数据库中没有可用的表情包，搜索功能暂不可用。")
            self._embedding_matrix = None
            self._all_hashes = []
            return

        self._all_hashes = [s.sticker_hash for s in valid_stickers]
        embeddings = [s.embedding for s in valid_stickers]
        
        # 转换为矩阵
        matrix = np.array(embeddings, dtype=np.float32)
        
        # 计算模长，增加 epsilon (1e-9) 防止除以 0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._embedding_matrix = matrix / (norms + 1e-9)
        logger.info(f"搜索索引重建完毕，覆盖 {len(self._all_hashes)} 个向量。")
