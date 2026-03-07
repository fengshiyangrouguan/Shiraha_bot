# src/cortices/reading/cortex.py
import asyncio
import os
import re
from typing import Optional, TYPE_CHECKING

from src.cortices.base_cortex import BaseCortex
from src.cortices.reading.context import AgentReadingContext
from src.system.di.container import container

if TYPE_CHECKING:
    from src.common.database.database_manager import DatabaseManager

# 定义书籍文件存放路径
RAW_BOOKS_DIR = "data/book"
REGISTERED_BOOKS_DIR = "data/book_registered"

class ReadingCortex(BaseCortex):
    """
    ReadingCortex 负责：
    1. 自动发现、切片和注册新书。
    2. 为 Agent 提供读书工具，并管理 Agent 的内部阅读状态。
    """
    
    def __init__(self):
        super().__init__()
        self.db_manager: Optional["DatabaseManager"] = None
        # Agent 当前的阅读任务上下文。同一时间只进行一项阅读任务。
        self.agent_reading_task: Optional[AgentReadingContext] = None

    async def setup(self, world_model, config, cortex_manager):
        """Cortex 启动时，解析依赖并启动后台书籍处理任务。"""
        await super().setup(world_model, config, cortex_manager)
        self.db_manager = container.resolve("DatabaseManager")
        print("📚 ReadingCortex: 启动，准备扫描新书...")
        asyncio.create_task(self._scan_and_process_new_books())

    async def _scan_and_process_new_books(self):
        """扫描原始书籍目录，处理新书并注册到数据库。"""
        os.makedirs(RAW_BOOKS_DIR, exist_ok=True)
        os.makedirs(REGISTERED_BOOKS_DIR, exist_ok=True)
        await asyncio.sleep(1)

        registered_books = await self.db_manager.get_all_books()
        registered_titles = {book.title for book in registered_books}
        print(f"📚 已注册书籍: {registered_titles or '无'}")

        for filename in os.listdir(RAW_BOOKS_DIR):
            book_title, _ = os.path.splitext(filename)
            if book_title not in registered_titles:
                await self._process_one_book(filename, book_title)

    async def _process_one_book(self, filename: str, book_title: str):
        """处理单本新书的切片和注册。"""
        print(f"📚 发现新书: {book_title}，正在处理...")
        raw_path = os.path.join(RAW_BOOKS_DIR, filename)
        registered_path = os.path.join(REGISTERED_BOOKS_DIR, f"{book_title}_tagged.txt")
        
        try:
            # 1. 内化的切片逻辑
            self._slice_and_tag_book(
                input_path=raw_path,
                output_path=registered_path,
                max_chunk_size=1000
            )
            # 2. 注册到数据库
            await self.db_manager.add_book(
                title=book_title,
                raw_file_path=raw_path,
                registered_file_path=registered_path,
                status="registered"
            )
            print(f"✅ 书籍 '{book_title}' 已成功切片并注册。")
        except Exception as e:
            print(f"❌ 处理书籍 '{book_title}' 时发生错误: {e}")
            await self.db_manager.add_book(
                title=book_title, raw_file_path=raw_path, 
                registered_file_path="", status="error"
            )

    def _slice_and_tag_book(self, input_path: str, output_path: str, max_chunk_size: int):
        """
        内化到 Cortex 中的书籍切片与标记功能。
        逻辑源自之前的 slice_and_tag_by_newline.py 脚本。
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        processed_chunks = []
        current_pos = 0
        text_len = len(text)
        tag_format = ("""

---(segment {num})---

""")

        while current_pos < text_len:
            search_from = current_pos + max_chunk_size
            if search_from >= text_len:
                processed_chunks.append(text[current_pos:])
                break
            
            split_pos = text.find('\n', search_from)
            if split_pos == -1:
                processed_chunks.append(text[current_pos:])
                break
            
            chunk = text[current_pos : split_pos]
            processed_chunks.append(chunk)
            current_pos = split_pos + 1

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(processed_chunks):
                f.write(chunk)
                if i < len(processed_chunks) - 1:
                    f.write(tag_format.format(num=i + 1))
        
        print(f"切分完成: {output_path}, 共 {len(processed_chunks)} 个片段。")

    async def teardown(self):
        print("📚 ReadingCortex: 正在关闭...")
