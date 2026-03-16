import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from json_repair import repair_json
from sqlalchemy import asc, select

from src.common.database.database_manager import DatabaseManager
from src.common.database.database_model import EventDB
from src.common.logger import get_logger
from src.llm_api.factory import LLMRequestFactory
from src.memory_system.models.expression_pattern import ExpressionPattern
from src.memory_system.repositories.expression_pattern_repository import ExpressionPatternRepository
from src.memory_system.services.expression_utils import calculate_similarity, filter_message_content

logger = get_logger("expressor")


LEARN_STYLE_PROMPT = """
{chat_str}

请从上面这段群聊中概括除了人名为"SELF"之外的人的语言风格。
1. 只考虑文字，不要考虑表情包和图片。
2. 不要涉及具体的人名，但是可以涉及具体名词。
3. 思考有没有特殊的梗，一并总结成语言风格。
4. 严格根据群聊内容总结。

请尽量按下面这种格式输出多条：
当"某个场景"时，使用"某种表达方式"
"""


MATCH_EXPRESSION_CONTEXT_PROMPT = """
聊天内容：
{chat_str}

表达方式 pairs：
{expression_pairs}

请为上面的每一条表达方式找到最匹配的原文句子，返回 JSON 数组。
每项格式：
{{
  "expression_pair": "序号",
  "context": "对应原文"
}}
"""


class ExpressionLearningService:
    def __init__(self, database_manager: DatabaseManager, llm_factory: LLMRequestFactory):
        self.database_manager = database_manager
        self.repository = ExpressionPatternRepository(database_manager)
        self.express_learn_model = llm_factory.get_request("utils")
        self.summary_model = llm_factory.get_request("utils_small")

    async def learn_from_recent_events(
        self,
        conversation_id: str,
        limit: int = 25,
        bot_user_id: Optional[str] = None,
    ) -> List[ExpressionPattern]:
        async with await self.database_manager.get_session() as session:
            stmt = (
                select(EventDB)
                .where(EventDB.conversation_id == conversation_id, EventDB.event_type == "message")
                .order_by(asc(EventDB.time))
                .limit(limit)
            )
            events = (await session.execute(stmt)).scalars().all()

        return await self.learn_from_messages(conversation_id, events, bot_user_id=bot_user_id)

    async def learn_from_messages(
        self,
        conversation_id: str,
        messages: List[Any],
        bot_user_id: Optional[str] = None,
    ) -> List[ExpressionPattern]:
        normalized_messages = self._normalize_messages(messages)
        logger.info(
            f"[ExpressionLearn] conversation_id={conversation_id}, "
            f"raw_message_count={len(messages)}, normalized_count={len(normalized_messages)}"
        )
        if not normalized_messages:
            logger.info(f"[ExpressionLearn] no normalized messages for {conversation_id}")
            return []

        chat_str = self._build_anonymous_messages(normalized_messages, bot_user_id=bot_user_id)
        bare_chat_str = self._build_bare_messages(normalized_messages)
        prompt = LEARN_STYLE_PROMPT.format(chat_str=chat_str)

        try:
            response, _ = await self.express_learn_model.execute(prompt, temperature=0.3)
            logger.info(f"[ExpressionLearn] raw style extraction output:\n{response}")
        except Exception as exc:
            logger.error(f"[ExpressionLearn] style extraction failed: {exc}")
            return []

        expressions = self.parse_expression_response(response)
        expressions = self._filter_self_reference_styles(expressions)
        logger.info(
            "[ExpressionLearn] parsed expressions:\n"
            + ("\n".join([f"- situation={s} | style={st}" for s, st in expressions]) if expressions else "- none")
        )
        if not expressions:
            return []

        matched_expressions = await self.match_expression_context(expressions, bare_chat_str)
        logger.info(
            "[ExpressionLearn] matched expressions:\n"
            + (
                "\n".join([f"- situation={s} | style={st} | context={ctx}" for s, st, ctx in matched_expressions])
                if matched_expressions
                else "- none"
            )
        )

        bare_lines = self._build_bare_lines(normalized_messages)
        learnt_patterns: List[ExpressionPattern] = []
        current_time = time.time()

        for situation, style, context in matched_expressions:
            pos = None
            for index, (_, content) in enumerate(bare_lines):
                if calculate_similarity(content, context) >= 0.85:
                    pos = index
                    break

            if pos is None or pos == 0:
                logger.info(
                    f"[ExpressionLearn] skip pair because no valid previous message: "
                    f"situation={situation}, style={style}"
                )
                continue

            up_original_idx = bare_lines[pos - 1][0]
            up_content = filter_message_content(normalized_messages[up_original_idx]["content"])
            if not up_content:
                logger.info(
                    f"[ExpressionLearn] skip pair because up_content is empty: "
                    f"situation={situation}, style={style}"
                )
                continue

            existing = await self.repository.get_by_style(conversation_id, style)
            if existing is None:
                logger.info(
                    f"[ExpressionLearn] create pattern: conversation_id={conversation_id}, "
                    f"situation={situation}, style={style}, up_content={up_content}"
                )
                await self.repository.create_pattern(
                    conversation_id=conversation_id,
                    situation=situation,
                    style=style,
                    context=context,
                    up_content=up_content,
                    current_time=current_time,
                )
                learnt_patterns.append(
                    ExpressionPattern(
                        chat_id=conversation_id,
                        situation=situation,
                        style=style,
                        count=1,
                        last_active_time=current_time,
                        create_date=current_time,
                        content_list=[situation],
                        context=context,
                        up_content=up_content,
                    )
                )
                continue

            content_list = self.repository._parse_content_list(existing.content_list)
            content_list.append(situation)
            summarized = await self._compose_situation_text(content_list, existing.count + 1, existing.situation)
            logger.info(
                f"[ExpressionLearn] update pattern id={existing.id}, old_situation={existing.situation}, "
                f"new_situation={summarized}, style={style}, up_content={up_content}"
            )
            await self.repository.update_existing_pattern(
                pattern_id=existing.id,
                situation=situation,
                context=context,
                up_content=up_content,
                current_time=current_time,
                summarized_situation=summarized,
            )
            learnt_patterns.append(
                ExpressionPattern(
                    chat_id=conversation_id,
                    situation=summarized,
                    style=style,
                    count=(existing.count or 0) + 1,
                    last_active_time=current_time,
                    create_date=existing.create_date,
                    content_list=content_list,
                    context=context,
                    up_content=up_content,
                    id=existing.id,
                )
            )

        await self.repository.limit_max_patterns(conversation_id, max_count=300)
        logger.info(f"[ExpressionLearn] learned pattern count for {conversation_id}: {len(learnt_patterns)}")
        return learnt_patterns

    async def match_expression_context(
        self,
        expression_pairs: List[Tuple[str, str]],
        bare_chat_str: str,
    ) -> List[Tuple[str, str, str]]:
        numbered_pairs = [
            f'{index}. 当"{situation}"时，使用"{style}"'
            for index, (situation, style) in enumerate(expression_pairs, 1)
        ]
        prompt = MATCH_EXPRESSION_CONTEXT_PROMPT.format(
            chat_str=bare_chat_str,
            expression_pairs="\n".join(numbered_pairs),
        )

        try:
            response, _ = await self.express_learn_model.execute(prompt, temperature=0.3)
            logger.info(f"[ExpressionLearn] raw context match output:\n{response}")
        except Exception as exc:
            logger.error(f"[ExpressionLearn] context match failed: {exc}")
            return []

        parsed = self._parse_match_response(response)
        matched_expressions: List[Tuple[str, str, str]] = []
        used_pair_indices = set()
        for item in parsed:
            try:
                pair_index = int(item["expression_pair"]) - 1
            except (KeyError, TypeError, ValueError):
                continue

            if not 0 <= pair_index < len(expression_pairs) or pair_index in used_pair_indices:
                continue

            situation, style = expression_pairs[pair_index]
            context = str(item.get("context", "")).strip()
            if not context:
                continue

            matched_expressions.append((situation, style, context))
            used_pair_indices.add(pair_index)
        return matched_expressions

    def parse_expression_response(self, response: str) -> List[Tuple[str, str]]:
        expressions: List[Tuple[str, str]] = []
        pattern = re.compile(r'当"(.*?)"时.*?使用"(.*?)"')
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            match = pattern.search(line)
            if match:
                situation = match.group(1).strip()
                style = match.group(2).strip()
                if situation and style:
                    expressions.append((situation, style))
        return expressions

    async def _compose_situation_text(self, content_list: List[str], count: int, fallback: str) -> str:
        sanitized = [item.strip() for item in content_list if item and item.strip()]
        summary = await self._summarize_situations(sanitized)
        if summary:
            return summary
        return "/".join(sanitized) if sanitized else fallback

    async def _summarize_situations(self, situations: List[str]) -> Optional[str]:
        if not situations:
            return None

        prompt = (
            "请将下面这些相似的表达场景归纳为一个更简洁、更概括的场景描述，"
            "不超过20个字，只输出归纳结果。\n"
            + "\n".join(f"- {item}" for item in situations[-10:])
        )
        try:
            summary, _ = await self.summary_model.execute(prompt, temperature=0.2)
            summary = summary.strip()
            logger.info(f"[ExpressionLearn] summarized situation: {summary}")
            return summary or None
        except Exception as exc:
            logger.error(f"[ExpressionLearn] summarize situations failed: {exc}")
            return None

    def _filter_self_reference_styles(self, expressions: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        banned = {"self", "SELF"}
        return [(situation, style) for situation, style in expressions if style.strip() not in banned]

    def _normalize_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for message in messages:
            if isinstance(message, dict):
                normalized.append(
                    {
                        "sender_id": str(message.get("sender_id") or message.get("user_id") or ""),
                        "content": str(message.get("content") or message.get("message") or "").strip(),
                        "timestamp": float(message.get("timestamp") or message.get("time") or time.time()),
                    }
                )
                continue

            normalized.append(
                {
                    "sender_id": str(getattr(message, "user_id", getattr(message, "sender_id", "")) or ""),
                    "content": str(
                        getattr(message, "event_content", getattr(message, "processed_plain_text", getattr(message, "content", "")))
                        or ""
                    ).strip(),
                    "timestamp": float(getattr(message, "time", getattr(message, "timestamp", time.time())) or time.time()),
                }
            )
        return [item for item in normalized if item["content"]]

    def _build_anonymous_messages(self, messages: List[Dict[str, Any]], bot_user_id: Optional[str]) -> str:
        lines: List[str] = []
        for item in messages:
            speaker = "SELF" if bot_user_id and item["sender_id"] == str(bot_user_id) else "OTHER"
            content = filter_message_content(item["content"])
            if content:
                lines.append(f"[{speaker}] {content}")
        return "\n".join(lines)

    def _build_bare_messages(self, messages: List[Dict[str, Any]]) -> str:
        return "\n".join(
            [content for content in [filter_message_content(item["content"]) for item in messages] if content]
        )

    def _build_bare_lines(self, messages: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
        return [(index, filter_message_content(item["content"])) for index, item in enumerate(messages)]

    def _parse_match_response(self, response: str) -> List[Dict[str, Any]]:
        response = response.strip()
        if not response:
            return []
        try:
            repaired = repair_json(response)
            data = json.loads(repaired) if isinstance(repaired, str) else repaired
        except Exception:
            return []
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    async def format_expression_patterns_for_prompt(self, conversation_id: str, limit: int = 10) -> str:
        patterns = await self.repository.get_patterns(conversation_id, limit=limit)
        if not patterns:
            logger.info(f"[ExpressionLearn] no stored patterns for prompt: {conversation_id}")
            return ""

        lines = ["以下是近期学到的表达方式："]
        for pattern in patterns:
            lines.append(f"- 当{pattern.situation}时，可参考「{pattern.style}」")
        formatted = "\n".join(lines)
        logger.info(f"[ExpressionLearn] formatted stored patterns block:\n{formatted}")
        return formatted
