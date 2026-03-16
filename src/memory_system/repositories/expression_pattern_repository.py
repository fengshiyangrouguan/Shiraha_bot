import time
import json
from typing import Dict, List, Optional

from sqlalchemy import delete, desc, select

from src.common.database.database_manager import DatabaseManager
from src.common.database.database_model import ExpressionPatternDB
from src.memory_system.models.expression_pattern import ExpressionPattern
from src.memory_system.services.expression_utils import weighted_sample


class ExpressionPatternRepository:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    async def get_by_style(self, conversation_id: str, style: str) -> Optional[ExpressionPatternDB]:
        async with await self.database_manager.get_session() as session:
            stmt = select(ExpressionPatternDB).where(
                ExpressionPatternDB.chat_id == conversation_id,
                ExpressionPatternDB.style == style,
            )
            return (await session.execute(stmt)).scalar_one_or_none()

    async def create_pattern(
        self,
        conversation_id: str,
        situation: str,
        style: str,
        context: str,
        up_content: str,
        current_time: float,
    ) -> None:
        async with await self.database_manager.get_session() as session:
            session.add(
                ExpressionPatternDB(
                    chat_id=conversation_id,
                    situation=situation,
                    style=style,
                    content_list=json.dumps([situation], ensure_ascii=False),
                    count=1,
                    last_active_time=current_time,
                    create_date=current_time,
                    context=context,
                    up_content=up_content,
                )
            )
            await session.commit()

    async def update_existing_pattern(
        self,
        pattern_id: int,
        situation: str,
        context: str,
        up_content: str,
        current_time: float,
        summarized_situation: str,
    ) -> None:
        async with await self.database_manager.get_session() as session:
            row = await session.get(ExpressionPatternDB, pattern_id)
            if row is None:
                return

            content_list = self._parse_content_list(row.content_list)
            content_list.append(situation)

            row.content_list = json.dumps(content_list, ensure_ascii=False)
            row.count = (row.count or 0) + 1
            row.last_active_time = current_time
            row.context = context
            row.up_content = up_content
            row.situation = summarized_situation
            await session.commit()

    async def get_patterns(self, conversation_id: str, limit: int = 20) -> List[ExpressionPattern]:
        async with await self.database_manager.get_session() as session:
            stmt = (
                select(ExpressionPatternDB)
                .where(ExpressionPatternDB.chat_id == conversation_id)
                .order_by(desc(ExpressionPatternDB.last_active_time), desc(ExpressionPatternDB.count))
                .limit(limit)
            )
            rows = (await session.execute(stmt)).scalars().all()

        return [self._to_pattern(row) for row in rows]

    async def random_patterns(self, conversation_id: str, total_num: int) -> List[Dict]:
        async with await self.database_manager.get_session() as session:
            stmt = select(ExpressionPatternDB)
            #TODO:改为coversation_id

            # stmt = select(ExpressionPatternDB).where(ExpressionPatternDB.chat_id == conversation_id)
            rows = (await session.execute(stmt)).scalars().all()

        population = [
            {
                "id": row.id,
                "situation": row.situation,
                "style": row.style,
                "last_active_time": row.last_active_time,
                "source_id": row.chat_id,
                "create_date": row.create_date,
                "count": row.count if row.count is not None else 1,
            }
            for row in rows
        ]
        return weighted_sample(population, total_num)

    async def update_last_active_time(self, expressions_to_update: List[Dict]) -> None:
        if not expressions_to_update:
            return

        now_ts = time.time()
        unique_keys = {}
        for expr in expressions_to_update:
            source_id = expr.get("source_id")
            situation = expr.get("situation")
            style = expr.get("style")
            if source_id and situation and style:
                unique_keys[(source_id, situation, style)] = True

        async with await self.database_manager.get_session() as session:
            for conversation_id, situation, style in unique_keys:
                stmt = select(ExpressionPatternDB).where(
                    ExpressionPatternDB.chat_id == conversation_id,
                    ExpressionPatternDB.situation == situation,
                    ExpressionPatternDB.style == style,
                )
                row = (await session.execute(stmt)).scalar_one_or_none()
                if row is not None:
                    row.last_active_time = now_ts
            await session.commit()

    async def limit_max_patterns(self, conversation_id: str, max_count: int = 300) -> None:
        async with await self.database_manager.get_session() as session:
            stmt = (
                select(ExpressionPatternDB)
                .where(ExpressionPatternDB.chat_id == conversation_id)
                .order_by(desc(ExpressionPatternDB.count), desc(ExpressionPatternDB.last_active_time))
            )
            rows = (await session.execute(stmt)).scalars().all()
            if len(rows) <= max_count:
                return

            delete_ids = [row.id for row in rows[max_count:] if row.id is not None]
            if delete_ids:
                await session.execute(delete(ExpressionPatternDB).where(ExpressionPatternDB.id.in_(delete_ids)))
                await session.commit()

    def _to_pattern(self, row: ExpressionPatternDB) -> ExpressionPattern:
        return ExpressionPattern(
            id=row.id,
            chat_id=row.chat_id,
            situation=row.situation,
            style=row.style,
            count=row.count,
            last_active_time=row.last_active_time,
            create_date=row.create_date,
            content_list=self._parse_content_list(row.content_list),
            context=row.context or "",
            up_content=row.up_content or "",
        )

    @staticmethod
    def _parse_content_list(raw_value: Optional[str]) -> List[str]:
        if not raw_value:
            return []
        try:
            data = json.loads(raw_value)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        return [str(item) for item in data if isinstance(item, str)]
