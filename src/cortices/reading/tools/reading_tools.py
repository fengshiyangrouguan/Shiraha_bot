# src/cortices/reading/tools/reading_tools.py
import re
from pydantic import BaseModel
from typing import TYPE_CHECKING, Optional

from src.cortices.tools_base import BaseTool
from src.cortices.reading.context import AgentReadingContext

if TYPE_CHECKING:
    from src.cortices.reading.cortex import ReadingCortex
    from src.common.database.database_manager import DatabaseManager

class StartOrContinueReadingTool(BaseTool):
    """
    Agent 使用此工具来启动或继续一项阅读任务。
    它会返回书籍的下一个片段供 Agent “阅读”和“思考”。
    """
    name: str = "start_or_continue_reading"
    description: str = "启动一本新书的阅读任务，或继续当前正在阅读的书。返回下一段内容。"
    
    class Parameters(BaseModel):
        book_title: Optional[str] = None
        
    def __init__(self, cortex: 'ReadingCortex', db_manager: 'DatabaseManager'):
        self.cortex = cortex
        self.db_manager = db_manager

    async def _run(self, params: Parameters) -> str:
        book_title = params.book_title
        
        # 如果指定了书名，则开始或切换到这本书
        if book_title:
            book = await self.db_manager.get_book_by_title(book_title)
            if not book:
                return f"错误: 书库里没有找到《{book_title}》。"
            
            try:
                with open(book.registered_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                chunks = re.split(r'\s*---(segment \d+)---\s*', content)
                chunks = [chunk for chunk in chunks if chunk.strip()]
            except Exception as e:
                return f"错误: 读取《{book_title}》文件时出错: {e}"

            # 创建新的阅读上下文
            context = AgentReadingContext(
                book_id=book.id,
                book_title=book.title,
                chunks=chunks
            )
            self.cortex.agent_reading_task = context
        else:
            # 如果未指定书名，则继续当前的阅读任务
            context = self.cortex.agent_reading_task
            if not context:
                return "错误: 当前没有正在进行的阅读任务，请指定 book_title 来开始。"

        # 获取当前片段
        chunk_content = context.get_current_chunk()
        
        if chunk_content is None:
            self.cortex.agent_reading_task = None # 阅读完毕，清空任务
            return f"你已经读完了《{context.book_title}》的全部内容。"
            
        return f"这是《{context.book_title}》的第 {context.current_chunk_index + 1} 段：

{chunk_content}"


class RecordReflectionTool(BaseTool):
    """
    Agent 在阅读并思考完一个片段后，使用此工具记录其总结和感想。
    """
    name: str = "record_reading_reflection"
    description: str = "记录对一个书籍片段的总结和感想，并决定是继续阅读还是退出。"
    
    class Parameters(BaseModel):
        summary: str
        reflection: str
        decision: str # "continue" 或 "exit"

    def __init__(self, cortex: 'ReadingCortex', db_manager: 'DatabaseManager'):
        self.cortex = cortex
        self.db_manager = db_manager

    async def _run(self, params: Parameters) -> str:
        context = self.cortex.agent_reading_task
        if not context:
            return "错误: 没有正在进行的阅读任务，无法记录感想。"

        # 1. 保存读后感到数据库
        await self.db_manager.add_reflection(
            book_id=context.book_id,
            chunk_index=context.current_chunk_index,
            summary=params.summary,
            reflection=params.reflection
        )
        
        # 2. 将感想存入短期记忆
        memory_entry = f"在读《{context.book_title}》第{context.current_chunk_index + 1}段时，我想到：{params.reflection}"
        context.short_term_memory.append(memory_entry)

        # 3. 根据 Agent 的决定推进或结束任务
        if params.decision.lower() == "continue":
            context.advance_to_next_chunk()
            return f"感想已记录。准备阅读《{context.book_title}》的下一段。"
        else: # exit
            self.cortex.agent_reading_task = None # 清空当前阅读任务
            return f"好的，感想已记录。我们暂停阅读《{context.book_title}》。"
