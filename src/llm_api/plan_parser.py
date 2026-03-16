# src/llm_api/plan_parser.py
import json
from typing import Optional

from src.common.logger import get_logger
from src.agent.planner.planner_result import PlanResult
from src.common.action_model.action_spec import ActionSpec

class PlanParser:
    """
    负责解析来自 LLM 的规划结果。
    将原始的、可能不规范的文本，转换为结构化的 PlanResult 对象。
    """
    def __init__(self, logger_name: str = "PlanParser"):
        self.logger = get_logger(logger_name)

    def parse(self, content: str) -> Optional[PlanResult]:
        """
        解析 LLM 返回的原始字符串。

        Args:
            content: LLM 返回的原始文本内容。

        Returns:
            一个 PlanResult 对象，如果解析或校验失败，则返回 None。
        """
        if not content:
            self.logger.error("LLM 未返回任何内容。")
            return None

        self.logger.debug(f"LLM 原始规划内容: {content}")

        # 尝试解析 JSON
        try:
            # 去除可能的 markdown 代码块标记
            cleaned_content = content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            
            parsed = json.loads(cleaned_content.strip())
        except json.JSONDecodeError:
            self.logger.error(f"LLM 返回内容不是合法的 JSON: {content}")
            return None

        # --- 校验 JSON 结构 ---
        required_top_fields = ["reason", "action"]
        missing_top = [k for k in required_top_fields if k not in parsed]

        if missing_top:
            self.logger.error(f"JSON 缺少顶层必要字段: {missing_top}")
            return None

        # 校验 action 内部结构
        if not isinstance(parsed.get("action"), dict):
            self.logger.error("JSON 中的 'action' 字段必须是一个字典。")
            return None
            
        required_action_fields = ["tool_name", "parameters"]
        missing_action = [k for k in required_action_fields if k not in parsed["action"]]

        if missing_action:
            self.logger.error(f"action 字段缺少必要字段: {missing_action}")
            return None

        # --- 字段类型检查 ---
        if not isinstance(parsed["reason"], str):
            self.logger.error(f"字段 'reason' 必须是字符串。")
            return None

        if not isinstance(parsed["action"]["tool_name"], str):
            self.logger.error(f"字段 'tool_name' 必须是字符串。")
            return None

        if not isinstance(parsed["action"]["parameters"], dict):
            self.logger.error(f"字段 'parameters' 必须是一个字典。")
            return None

        # --- 构造 ActionSpec 和 PlanResult ---
        try:
            action_spec = ActionSpec(
                tool_name=parsed["action"]["tool_name"],
                parameters=parsed["action"]["parameters"],
                action_type=parsed["action"].get("action_type", "tool"),
                source="main_planner",
            )

            plan_result = PlanResult(
                reason=parsed["reason"],
                action=action_spec
            )

            self.logger.info(f"思考过程: {plan_result.reason}")
            self.logger.info(f"计划行动: 调用工具 '{plan_result.action.tool_name}' 参数: {plan_result.action.parameters}")

            return plan_result

        except Exception as e:
            self.logger.error(f"构造 PlanResult 时发生未知错误: {e}")
            return None
