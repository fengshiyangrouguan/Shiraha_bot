from typing import Any, Dict, List

from src.agent.world_model import WorldModel
from src.common.logger import get_logger
from src.cortex_system.tools_base import BaseTool
from src.cortices.reading.reading_data import ReadingData

logger = get_logger("reading_atomic_tools")


class ViewBookshelfTool(BaseTool):
    """查看书架面板。"""

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    @property
    def scope(self) -> List[str]:
        return ["reading"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "view_bookshelf",
            "description": "查看当前书架与阅读进度。",
            "parameters": {},
            "required": [],
            "tool_kind": "panel",
        }

    async def execute(self, **kwargs) -> Any:
        reading_data: ReadingData = await self.world_model.get_cortex_data("reading_data")
        if not reading_data or not reading_data.book_dict:
            return {"panel": "bookshelf", "items": []}

        items = []
        for book in reading_data.book_dict.values():
            items.append(
                {
                    "book_title": book.book_title,
                    "status": book.status,
                    "current_chunk_index": book.current_chunk_index,
                    "total_chunks": book.total_chunks,
                    "is_finished_reading": book.is_finished_reading,
                }
            )
        return {"panel": "bookshelf", "items": items}


class ReadBookChunkTool(BaseTool):
    """读取一本书的一个小片段。"""

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    @property
    def scope(self) -> List[str]:
        return ["reading"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "read_book_chunk",
            "description": "读取指定书籍当前片段，并推进一个小步长。",
            "parameters": {
                "book_title": {"type": "string", "description": "书名"},
            },
            "required": ["book_title"],
            "tool_kind": "action",
        }

    async def execute(self, **kwargs) -> Any:
        book_title = kwargs.get("book_title", "")
        reading_data: ReadingData = await self.world_model.get_cortex_data("reading_data")
        if not reading_data or book_title not in reading_data.book_dict:
            return {"success": False, "error": f"未找到书籍: {book_title}"}

        book = reading_data.book_dict[book_title]
        if not book.chunks and book.registered_file_path:
            with open(book.registered_file_path, "r", encoding="utf-8") as file:
                book.chunks = file.read().split("\n\n---(segment)---\n\n")
            if not book.total_chunks:
                book.total_chunks = len(book.chunks)

        if not book.chunks:
            return {"success": False, "error": f"书籍没有可读内容: {book_title}"}

        index = min(book.current_chunk_index or 0, max(len(book.chunks) - 1, 0))
        content = book.chunks[index]
        if index < len(book.chunks) - 1:
            book.current_chunk_index = index + 1
            book.status = "在读"
        else:
            book.current_chunk_index = index
            book.is_finished_reading = True
            book.status = "已读完"

        await self.world_model.save_cortex_data("reading_data", reading_data)
        return {
            "success": True,
            "book_title": book_title,
            "chunk_index": index,
            "content": content,
            "finished": book.is_finished_reading,
        }


class MarkBookDormantTool(BaseTool):
    """把书标记到暂不阅读区域。"""

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    @property
    def scope(self) -> List[str]:
        return ["reading"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "mark_book_dormant",
            "description": "将某本书标记为暂不阅读。",
            "parameters": {
                "book_title": {"type": "string", "description": "书名"},
            },
            "required": ["book_title"],
            "tool_kind": "action",
        }

    async def execute(self, **kwargs) -> Any:
        book_title = kwargs.get("book_title", "")
        reading_data: ReadingData = await self.world_model.get_cortex_data("reading_data")
        if not reading_data or book_title not in reading_data.book_dict:
            return {"success": False, "error": f"未找到书籍: {book_title}"}

        book = reading_data.book_dict[book_title]
        book.status = "暂不阅读"
        await self.world_model.save_cortex_data("reading_data", reading_data)
        return {"success": True, "book_title": book_title, "status": book.status}
