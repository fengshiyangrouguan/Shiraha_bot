# src/cortices/reading/context.py
from collections import deque
from typing import List, Deque, Dict, Optional
from pydantic import BaseModel, Field
from src.common.database.database_manager import DatabaseManager
from src.common.database.database_model import BookDB
from time import time
from src.common.logger import get_logger

logger = get_logger("reading")
class Book(BaseModel):
    """书籍模型"""
    book_title: str
    # 书籍的全部内容，已切分成片段
    format: str
    registered_file_path:str
    status:str
    last_read_time: Optional[float] = None
    is_finished_reading: Optional[bool] = False
    chunks: Optional[List[str]] = Field(default_factory=list)
    total_chunks: Optional[int] = None
    current_chunk_index: Optional[int] = 0

    @property
    def total_chunks(self) -> int:
        """计算总片段数。"""
        return self.total_chunks

    def get_current_chunk(self) -> str:
        """获取当前要阅读的片段。"""
        if 0 <= self.current_chunk_index < self.total_chunks:
            return self.chunks[self.current_chunk_index]
        return None # 返回 None 表示已读完

    def advance_to_next_chunk(self):
        """将索引推进到下一个片段。"""
        self.current_chunk_index += 1



class ReadingData(BaseModel):
    """
    用于在内存中维护 Agent 自己的阅读任务状态。
    """
    book_dict: Dict[str, Book] = Field(default_factory=dict)  # 可选：存储书籍字典，供 Agent 选择阅读
    current_reading_book: Book = None  # 当前正在阅读的书籍对象
    short_term_memory: Deque[str] = Field(default_factory=lambda: deque(maxlen=10))

    def init_library(self, registered_books: List[Book]):
        """更新书籍列表，供 Agent 选择阅读。"""
        self.book_dict = {book.book_title: book for book in registered_books}

    def update_library(self, new_book: Book):
        """添加新书到书籍列表中。"""
        self.book_dict[new_book.book_title] = new_book

    def get_book_titles(self) -> List[str]:
        """从已加载的书籍对象中提取所有书名。"""
        return list(self.book_dict.keys())
    
    def set_book_status(self,status:str):
        self.current_reading_book.status = status

    def set_book_finished(self):
        self.current_reading_book.is_finished_reading = True

    async def update_book_progress_to_db(self, db_manager: DatabaseManager):
        """
        将当前阅读 Book 对象的最新进度同步到数据库。
        """
        # 1. 查找数据库中对应的书籍记录
        # 注意：这里建议用 book_title 或 registered_file_path 作为唯一标识
        self.current_reading_book.last_read_time = time()

        db_book:BookDB = await db_manager.get(BookDB, self.current_reading_book.book_title)
        if db_book:
            # 2. 更新字段
            db_book.last_read_position = self.current_reading_book.current_chunk_index
            db_book.status = self.current_reading_book.status
            db_book.is_finished_reading = self.current_reading_book.current_chunk_index >= self.current_reading_book.total_chunks - 1
            db_book.last_read_time = self.current_reading_book.last_read_time
            
            # 3. 提交更新
            await db_manager.upsert(db_book)
            logger.info(f"更新《{self.current_reading_book.book_title}》 进度至 {self.current_reading_book.current_chunk_index}")
        else:
            logger.warning(f"中未找到书籍《{self.current_reading_book.book_title}》，无法保存进度。")
