# src/common/database/database_manager.py
from typing import Any, Optional, Type

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlmodel import SQLModel


class DatabaseManager:
    """
    数据库管理器。

    设计目标：
    1. 优先使用异步 SQLite 驱动，保持原项目接口不变。
    2. 如果运行环境缺少 `aiosqlite`，自动退回同步 SQLite。
    3. 对上层维持统一的 async API，避免主链因为环境依赖缺失直接启动失败。
    """

    def __init__(self):
        self._engine = None
        self._session_maker = None
        self._use_async_driver = True

    async def initialize_database(self, db_path: str = "data/shiraha_bot.db", echo: bool = False):
        """
        初始化数据库引擎并创建表结构。

        回退策略：
        - 能导入 `aiosqlite` 时使用异步引擎。
        - 否则回退到同步 sqlite 引擎，但外部接口依然保持 async 形式。
        """
        try:
            import aiosqlite  # noqa: F401
            self._use_async_driver = True
        except ModuleNotFoundError:
            self._use_async_driver = False

        from . import database_model  # noqa: F401

        if self._use_async_driver:
            db_url = f"sqlite+aiosqlite:///{db_path}"
            self._engine = create_async_engine(db_url, echo=echo)
            self._session_maker = sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            async with self._engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
        else:
            db_url = f"sqlite:///{db_path}"
            self._engine = create_engine(db_url, echo=echo)
            self._session_maker = sessionmaker(
                self._engine,
                class_=Session,
                expire_on_commit=False,
            )
            SQLModel.metadata.create_all(self._engine)

        print(f"数据库已初始化于: {db_url}")

    async def get_session(self):
        """
        获取数据库会话。

        注意：
        - 异步模式返回 AsyncSession。
        - 同步回退模式返回普通 Session，但依然通过 async 函数返回，
          这样调用方不需要为了环境差异改代码。
        """
        return self._session_maker()

    async def upsert(self, model_object: SQLModel):
        """
        插入或更新对象。
        """
        if self._use_async_driver:
            async with await self.get_session() as session:
                await session.merge(model_object)
                await session.commit()
            return

        session: Session = await self.get_session()
        with session:
            session.merge(model_object)
            session.commit()

    async def get(self, model_class: Type[SQLModel], pk: Any) -> Optional[SQLModel]:
        """
        根据主键读取对象。
        """
        if self._use_async_driver:
            async with await self.get_session() as session:
                result = await session.get(model_class, pk)
                return result

        session: Session = await self.get_session()
        with session:
            return session.get(model_class, pk)

    async def get_all(self, query):
        """
        执行查询并返回结果列表。
        """
        if self._use_async_driver:
            async with await self.get_session() as session:
                result = await session.execute(query)
                return [row[0] for row in result.all()]

        session: Session = await self.get_session()
        with session:
            result = session.execute(query)
            return [row[0] for row in result.all()]

    async def close(self):
        """兼容 MainSystem 的关闭调用。"""
        await self.shutdown()

    async def shutdown(self):
        """关闭数据库引擎。"""
        if not self._engine:
            return

        if self._use_async_driver and hasattr(self._engine, "dispose"):
            await self._engine.dispose()
        elif hasattr(self._engine, "dispose"):
            self._engine.dispose()

        self._engine = None
        print("数据库引擎已关闭。")
