import os
import asyncio
from datetime import datetime
from typing import Optional
from sqlalchemy import select  
from src.common.database.database_manager import DatabaseManager
from src.common.database.database_model import BookDB
from src.cortices.reading.reading_data import Book

def slice_and_tag_book(input_path: str, output_path: str, max_chunk_size: int) -> int :
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

---(segment)---

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
    return len(processed_chunks)

async def load_all_books(db_manager: DatabaseManager) -> list[BookDB]:
    """从数据库中加载所有已注册书籍的完整信息。"""
    stmt = select(BookDB)

    if getattr(db_manager, "_use_async_driver", True):
        async with await db_manager.get_session() as session:
            result = await session.execute(stmt)
            db_books = result.scalars().all()
    else:
        session = await db_manager.get_session()
        with session:
            result = session.execute(stmt)
            db_books = result.scalars().all()

    materialized_books = []
    for db_item in db_books:
        # 1. 读取切片文件内容 (假设你把 chunks 存在了本地 txt 里)
        # 如果你之前是用切片工具生成的 tagged.txt，这里需要解析它
        # 2. 转换为你的 Pydantic Book 对象
        book_obj = Book(
            book_title=db_item.book_title,
            format=db_item.format,
            registered_file_path=db_item.registered_file_path,
            status=db_item.status,
            last_read_time=db_item.last_read_time,
            is_finished_reading=db_item.is_finished_reading,
            total_chunks=db_item.total_chunks,
            current_chunk_index=db_item.last_read_position or 0
        )
        materialized_books.append(book_obj)
    return list(materialized_books)

async def save_book_to_db(db_manager: DatabaseManager, book_title: str, format: str, registered_file_path: str,status: str = "未读", last_read_position: int = 0, last_read_time: float = None, is_finished_reading: bool = False,total_chunks: Optional[int] = None):
    """将新书信息保存到数据库中。"""
    new_book_db = BookDB(
        book_title=book_title,
        format=format,
        registered_file_path=registered_file_path,
        status=status,
        last_read_position=last_read_position,
        last_read_time=last_read_time,
        total_chunks=total_chunks,
        is_finished_reading=is_finished_reading)

    if getattr(db_manager, "_use_async_driver", True):
        async with await db_manager.get_session() as session:
            session.add(new_book_db)
            await session.commit()
    else:
        session = await db_manager.get_session()
        with session:
            session.add(new_book_db)
            session.commit()
