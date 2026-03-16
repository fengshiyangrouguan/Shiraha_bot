import json
from typing import Any, Dict, List, Optional, Tuple

from json_repair import repair_json

from src.common.config.schemas.bot_config import BotConfig
from src.common.logger import get_logger
from src.llm_api.factory import LLMRequestFactory
from src.memory_system.repositories.expression_pattern_repository import ExpressionPatternRepository

logger = get_logger("expression_selector")


EXPRESSION_EVALUATION_PROMPT = """{chat_observe_info}

你的名字是{bot_name}{target_message}
{reply_reason_block}

以下是可选的表达情境：
{all_situations}

请你分析聊天内容的语境、情绪、话题类型，从上述情境中选择最适合当前聊天情境的，最多{max_num}个情境。
考虑因素包括：
1. 聊天的情绪氛围（轻松、严肃、幽默等）
2. 话题类型（日常、技术、游戏、情感等）
3. 情境与当前语境的匹配度
{target_message_extra_block}

请以 JSON 格式输出，只需要输出选中的情境编号：
{{
  "selected_situations": [2, 3, 5]
}}

不要输出其他内容。
"""


class ExpressionSelectorService:
    def __init__(
        self,
        repository: ExpressionPatternRepository,
        llm_factory: LLMRequestFactory,
        bot_config: BotConfig,
    ):
        self.repository = repository
        self.llm_model = llm_factory.get_request("utils_small")
        self.bot_name = bot_config.persona.bot_name

    async def select_suitable_expressions(
        self,
        conversation_id: str,
        chat_info: str,
        max_num: int = 10,
        target_message: Optional[str] = None,
        reply_reason: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        style_exprs = await self.repository.random_patterns(conversation_id, 20)
        logger.info(
            f"[表达选择] conversation_id={conversation_id}, "
            f"候选数量={len(style_exprs)}, 最大选择数量={max_num}, "
            f"是否有目标消息={bool(target_message)}, 是否有回复原因={bool(reply_reason)}"
        )

        if len(style_exprs) < 10:
            logger.info(
                f"[表达选择] 跳过选择，因为候选数量 < 10 "
                f"(conversation_id={conversation_id})"
            )
            return [], []

        all_expressions: List[Dict[str, Any]] = []
        all_situations: List[str] = []
        for expr in style_exprs:
            expr = expr.copy()
            all_expressions.append(expr)
            all_situations.append(
                f'{len(all_expressions)}. 当 {expr["situation"]} 时，使用 {expr["style"]}'
            )

        logger.info(
            "[表达选择] 采样到的候选:\n"
            + "\n".join(
                [
                    f"- {idx + 1}: 情境={expr['situation']} | 风格={expr['style']} | 次数={expr.get('count')}"
                    for idx, expr in enumerate(all_expressions)
                ]
            )
        )

        target_message_str = f"，当前目标消息是：{target_message}" if target_message else ""
        target_message_extra_block = "4. 注意目标消息与表达情境的贴合度" if target_message else ""
        reply_reason_block = f"回复原因：{reply_reason}" if reply_reason else ""
        chat_context = f"以下是当前聊天观察信息：\n{chat_info}" if not reply_reason else ""

        prompt = EXPRESSION_EVALUATION_PROMPT.format(
            chat_observe_info=chat_context,
            bot_name=self.bot_name,
            target_message=target_message_str,
            reply_reason_block=reply_reason_block,
            all_situations="\n".join(all_situations),
            max_num=max_num,
            target_message_extra_block=target_message_extra_block,
        )

        try:
            content, _ = await self.llm_model.execute(prompt)
            logger.info(f"[表达选择] LLM原始输出: {content}")
        except Exception as exc:
            logger.error(f"[表达选择] LLM执行失败: {exc}")
            return [], []

        try:
            repaired = repair_json(content)
            result = json.loads(repaired) if isinstance(repaired, str) else repaired
        except Exception:
            logger.error(f"[表达选择] 解析LLM输出失败: {content}")
            return [], []

        if not isinstance(result, dict) or "selected_situations" not in result:
            logger.info(f"[表达选择] LLM输出缺少selected_situations字段: {result}")
            return [], []

        valid_expressions: List[Dict[str, Any]] = []
        selected_ids: List[int] = []
        for idx in result["selected_situations"]:
            if isinstance(idx, int) and 1 <= idx <= len(all_expressions):
                expression = all_expressions[idx - 1]
                valid_expressions.append(expression)
                if expression.get("id") is not None:
                    selected_ids.append(expression["id"])

        logger.info(
            f"[表达选择] 选中的索引={result.get('selected_situations', [])}, "
            f"选中的ID={selected_ids}"
        )
        if valid_expressions:
            logger.info(
                "[表达选择] 选中的表达:\n"
                + "\n".join(
                    [
                        f"- 情境={expr['situation']} | 风格={expr['style']}"
                        for expr in valid_expressions
                    ]
                )
            )
            await self.repository.update_last_active_time(valid_expressions)
        else:
            logger.info(f"[表达选择] 没有为 {conversation_id} 选中有效的表达")

        return valid_expressions, selected_ids

    def format_selected_expressions_for_prompt(self, expressions: List[Dict[str, Any]]) -> str:
        if not expressions:
            logger.info("[表达选择] 没有要格式化到提示词中的表达")
            return ""

        lines = ["以下是可参考的表达方式："]
        for expr in expressions:
            lines.append(f'- 当{expr["situation"]}时，可以使用句式「{expr["style"]}」')

        formatted = "\n".join(lines)
        logger.info(f"[表达选择] 提示词表达区块:\n{formatted}")
        return formatted