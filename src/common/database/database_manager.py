# src/common/database/database_manager.py
from typing import Type, Optional, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel


class DatabaseManager:
    """
    数据库连接、会话及操作的管理器。
    本类的实例将在主应用程序中创建，并通过依赖注入（DI）容器进行注入。
    """
    def __init__(self):
        self._engine = None
        self._session_maker = None

    async def initialize_database(self, db_path: str = "data/shiraha_bot.db", echo: bool = False):
        """
        初始化异步数据库引擎并创建所有数据表。

        Args:
            db_path (str): 数据库文件的路径，默认为 "data/shiraha_bot.db"。
            echo (bool): 是否打印 SQL 语句，默认为 False。
        """
        # 构建数据库连接字符串
        db_url = f"sqlite+aiosqlite:///{db_path}"
        
        self._engine = create_async_engine(db_url, echo=echo)
        self._session_maker = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # # 在此处导入 models 以确保它们被 SQLModel 的元数据注册
        from . import database_model
        async with self._engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        
        print(f"数据库已初始化于: {db_url}")

    async def get_session(self) -> AsyncSession:
        """从会话工厂提供一个异步会话。"""
        return self._session_maker()

    async def upsert(self, model_object: SQLModel):
        """
        根据主键插入新对象或更新现有对象。
        """
        async with await self.get_session() as session:
            # merge() 操作会自动处理插入或更新。
            await session.merge(model_object)
            await session.commit()

    async def get(self, model_class: Type[SQLModel], pk: Any) -> Optional[SQLModel]:
        """
        根据主键检索对象。
        """
        async with await self.get_session() as session:
            result = await session.get(model_class, pk)
            return result

    async def shutdown(self):
        """关闭数据库引擎连接。"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            print("数据库引擎已关闭。")