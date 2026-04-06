# src/cortices/reading/cortex.py
import asyncio
import os
import re
from typing import Optional, TYPE_CHECKING,List

from src.cortex_system.base_cortex import BaseCortex
from src.cortex_system.tools_base import BaseTool
from src.cortices.reading.reading_data import ReadingData,Book
from src.cortices.reading.tools.atomic_tools import MarkBookDormantTool, ReadBookChunkTool, ViewBookshelfTool
from src.common.di.container import container
from src.cortices.reading.utils.book_file_process import slice_and_tag_book,load_all_books,save_book_to_db
from src.common.logger import get_logger
from src.common.database.database_manager import DatabaseManager
from src.llm_api.factory import LLMRequestFactory
from src.agent.world_model import WorldModel
    

# 定义书籍文件存放路径
RAW_BOOKS_DIR = "data/book"
REGISTERED_BOOKS_DIR = "data/book_registered"
logger = get_logger("reading")

class ReadingCortex(BaseCortex):
    """
    ReadingCortex 负责：
    1. 自动发现、切片和注册新书。
    2. 为 Agent 提供读书工具，并管理 Agent 的内部阅读状态。
    """
    
    def __init__(self):
        super().__init__()

        self.llm_request_factory: Optional[LLMRequestFactory] = None
        self.database_manager: Optional[DatabaseManager] = None
        self._world_model: Optional[WorldModel] = None

        # Agent 当前的阅读任务上下文。同一时间只进行一项阅读任务。
        self.reading_data = ReadingData()

    async def setup(self, config, signal_callback=None, skill_manager=None):
        """Cortex 启动时，解析依赖并启动后台书籍处理任务。"""
        await super().setup(config, signal_callback, skill_manager)
        self._world_model = container.resolve(WorldModel)
        self.database_manager = container.resolve(DatabaseManager)
        self.llm_request_factory = container.resolve(LLMRequestFactory)
        await self._world_model.save_cortex_data("reading_data", self.reading_data)  # 初始化时保存空的阅读上下文
        logger.info("ReadingData: 启动，开始扫描书架...")
        await self._load_registered_books_context()  # 启动时先加载已注册书籍的上下文

        asyncio.create_task(self._scan_and_process_new_books())
        # await self._scan_and_process_new_books()
    
    async def _load_registered_books_context(self):
        """
        从数据库加载已注册书籍的信息，初始化 ReadingData。
        这保证了 Agent 在扫描新书前，就已经知道自己‘读过什么’。
        """
        registered_books:List[Book] = await load_all_books(self.database_manager)
        
        # 将已注册书籍同步到 reading_data 中
        # 将最新的阅读上下文保存到 WorldModel
        self.reading_data.init_library(registered_books) 
        await self._world_model.save_cortex_data("reading_data", self.reading_data)

    async def _scan_and_process_new_books(self):
        """扫描原始书籍目录，处理新书并注册到数据库。"""
        os.makedirs(RAW_BOOKS_DIR, exist_ok=True)
        os.makedirs(REGISTERED_BOOKS_DIR, exist_ok=True)
        await asyncio.sleep(1)

        registered_titles = self.reading_data.get_book_titles()
        logger.info(f"已注册书籍: {registered_titles or '无'}")

        for filename in os.listdir(RAW_BOOKS_DIR):
            book_title, format = os.path.splitext(filename)
            # eg: .md→md, .txt→txt
            format = format.lstrip('.').lower()
            new_books_list = []
            if book_title not in registered_titles:
                new_books_list.append(book_title)
                await self._process_one_book(filename, book_title, format)
            new_books_str = ", ".join(new_books_list)
            self._world_model.notifications["书房"] = f"上架了新书：{new_books_str}"

    async def _process_one_book(self, filename: str, book_title: str, format: str):
        """处理单本新书的切片和注册。"""
        logger.info(f"📚 发现新书: {book_title}，正在处理...")
        raw_path = os.path.join(RAW_BOOKS_DIR, filename)
        registered_file_path = os.path.join(REGISTERED_BOOKS_DIR, f"{filename}")
        
        try:
            # 1. 内化的切片逻辑
            total_chunks =slice_and_tag_book(
                input_path=raw_path,
                output_path=registered_file_path,
                max_chunk_size=1000
            )
            # 2. 注册到数据库
            await save_book_to_db(
                db_manager=self.database_manager,
                book_title=book_title,
                format=format,
                registered_file_path=registered_file_path,
                status="新书未读",
                last_read_position=0,
                last_read_time=None,
                is_finished_reading=False,
                total_chunks=total_chunks
            )

            logger.info(f"书籍 '{book_title}' 已成功切片并注册。")
            new_book = Book(
                book_title=book_title,
                format=format,
                registered_file_path=registered_file_path,
                status="新书未读",
                last_read_time=None,
                is_finished_reading=False,
                total_chunks=total_chunks
            )
            self.reading_data.update_library(new_book)
            await self._world_model.save_cortex_data("reading_data", self.reading_data)
        except Exception as e:
            logger.info(f"处理书籍 '{book_title}' 时发生错误: {e}")
    

    async def get_cortex_summary(self) -> str:
        """
        获取书房皮层的实时感知摘要，直接展示书架书籍列表和阅读进度。
        """
        reading_data: ReadingData = await self._world_model.get_cortex_data("reading_data")
        
        # 1. 基础校验
        if not reading_data or not reading_data.book_dict:
            return "书架是空的欸，没有找到任何书。"

        summary_lines = ["书房状态概览"]

        #TODO 检查是否有新上架通知（来自扫描任务）

        # 3. 遍历书架详细信息 (同步 EnterLibraryTool 的逻辑)
        summary_lines.append("--- 书架列表 ---")
        for book in reading_data.book_dict.values():
            # 进度计算逻辑优化：防止除以 0，确保 total_chunks 已被初始化
            total = book.total_chunks if hasattr(book, 'total_chunks') else 0
            current = book.current_chunk_index if hasattr(book, 'current_chunk_index') else 0
            
            progress = (current / total * 100) if total > 0 else 0.0
            
            status_tag = book.status
            if getattr(book, 'is_finished_reading', False):
                status_tag = "已读完"

            book_info = f"- 《{book.book_title}》 [{status_tag}] | 进度: {progress:.1f}%"
            
            # 只有读过的书才显示时间
            if status_tag != "新书未读" and getattr(book, 'last_read_time', None):
                from datetime import datetime
                dt = datetime.fromtimestamp(book.last_read_time)
                book_info += f" | 上次阅读: {dt.strftime('%Y-%m-%d %H:%M')}"
            
            summary_lines.append(book_info)

        # 4. 当前正打开的书籍
        if reading_data.current_reading_book:
            summary_lines.append(f"\n当前案头正翻开的书: 《{reading_data.current_reading_book.book_title}》")

        return "\n".join(summary_lines)
        

    async def teardown(self):
        logger.info("ReadingCortex: 正在关闭...")

    def get_tools(self) -> List[BaseTool]:
        """
        暴露阅读域的原子动作与只读面板。
        """
        return [
            ViewBookshelfTool(world_model=self._world_model),
            ReadBookChunkTool(world_model=self._world_model),
            MarkBookDormantTool(world_model=self._world_model),
        ]
