import json
from typing import TYPE_CHECKING, Any, Dict, List

from src.agent.world_model import WorldModel
from src.common.action_model.action_spec import ActionSpec
from src.common.action_model.tool_result import ToolResult
from src.common.database.database_manager import DatabaseManager
from src.common.di.container import container
from src.common.logger import get_logger
from src.cortices.qq_chat.chat.replyer import QQReplyer
from src.cortices.qq_chat.cortex import QQChatCortex
from src.cortices.qq_chat.data_model.qq_chat_data import QQChatData
from src.cortices.tools_base import BaseTool
from src.llm_api.factory import LLMRequestFactory
from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter

if TYPE_CHECKING:
    from src.cortices.manager import CortexManager

logger = get_logger("quick_reply")


class QuickReplyTool(BaseTool):
    """轻量会话编排器：回复一轮后，再决定退出、继续 quick reply、转入 deep chat 或追加其他工具动作。"""

    def __init__(
        self,
        adapter: QQNapcatAdapter,
        cortex: QQChatCortex,
        cortex_manager: "CortexManager",
    ):
        super().__init__(cortex_manager)
        self._world_model = container.resolve(WorldModel)
        self._db_manager = container.resolve(DatabaseManager)
        self._llm_factory = container.resolve(LLMRequestFactory)
        self._replyer = QQReplyer(
            world_model=self._world_model,
            adapter=adapter,
            llm_request_factory=self._llm_factory,
            database_manager=self._db_manager,
            cortex=cortex,
        )

    @property
    def scope(self) -> List[str]:
        return ["qq_app"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "quick_reply",
            "description": "对指定 QQ 会话执行一轮轻量回复，并在结束后决定是退出、继续 quick reply、转入 deep chat，或调用该作用域下的其他工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "目标会话的 conversation_id。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "本轮快速回复的目标或原因。",
                    },
                    "turn_index": {
                        "type": "integer",
                        "description": "当前 quick reply 的轮次，从 1 开始。",
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "允许连续执行 quick reply 的最大轮次。",
                    },
                },
                "required": ["conversation_id", "reason"],
            },
        }

    def _get_follow_up_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = self.cortex_manager.get_tool_schemas(["quick_reply", "global"])
        filtered: List[Dict[str, Any]] = []
        for schema in schemas:
            function_def = schema.get("function", {})
            name = function_def.get("name")
            if name == "batch_quick_plan":
                continue
            filtered.append(schema)
        return filtered

    @staticmethod
    def _build_tool_schema_map(tool_schemas: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        schema_map: Dict[str, Dict[str, Any]] = {}
        for schema in tool_schemas:
            function_def = schema.get("function", {})
            name = function_def.get("name")
            if name:
                schema_map[name] = function_def
        return schema_map

    def _build_follow_up_prompt(
        self,
        conversation_name: str,
        reason: str,
        reply_summary: str,
        latest_history: str,
        turn_index: int,
        max_turns: int,
        tool_schemas: List[Dict[str, Any]],
    ) -> str:
        tool_schema_text = json.dumps(tool_schemas, ensure_ascii=False, indent=2)
        return f"""
你是一个轻量 QQ 会话处理器。你刚刚完成了一次 quick reply，现在需要判断下一步该怎么做。

会话名称: {conversation_name}
本轮回复目标: {reason}
当前轮次: {turn_index}/{max_turns}

本轮 quick reply 执行摘要:
{reply_summary}

最新会话记录:
{latest_history}

你现在有两类输出手段：
1. decision:
- exit: 当前话题已自然收束，直接退出。
- quick_reply: 还需要再来一轮轻量回复。
- deep_chat: 当前话题变复杂了，应交给 enter_deep_chat。
2. actions:
- 可以追加调用当前作用域下的其他工具。
- 如果需要先查信息、再做后续回复，就把工具的名字和参数写进 actions。
- actions 里的工具会先执行，然后 AgentLoop 再处理 decision 对应的后续动作。

可用工具列表:
{tool_schema_text}

决策要求:
- 如果 turn_index 已达到 max_turns，禁止选择 quick_reply。
- 只有在明显需要更深理解或更强兴趣时才选择 deep_chat。
- 如果无需其他工具，actions 返回空数组。
- 只选择当前工具列表中存在的工具。

只输出 JSON：
{{
  "decision": "exit | quick_reply | deep_chat",
  "reason": "给 quick_reply 或 deep_chat 的简短原因；如果 decision=exit 可以留空",
  "actions": [
    {{
      "action": "工具名",
      "parameters": {{
        "参数名": "参数值"
      }}
    }}
  ]
}}
"""

    @staticmethod
    def _parse_follow_up_decision(content: str) -> Dict[str, Any]:
        json_str = (content or "").strip()
        if json_str.startswith("```"):
            json_str = json_str.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            raise ValueError("follow-up 决策结果不是 JSON 对象")
        raw_actions = parsed.get("actions", [])
        if raw_actions is None:
            raw_actions = []
        if not isinstance(raw_actions, list):
            raise ValueError("actions 必须是数组")
        return {
            "decision": str(parsed.get("decision", "exit")).strip(),
            "reason": str(parsed.get("reason", "")).strip(),
            "actions": raw_actions,
        }

    def _normalize_tool_actions(
        self,
        raw_actions: List[Any],
        conversation_id: str,
        tool_schema_map: Dict[str, Dict[str, Any]],
    ) -> List[ActionSpec]:
        normalized_actions: List[ActionSpec] = []

        for raw_action in raw_actions:
            if not isinstance(raw_action, dict):
                continue

            tool_name = str(
                raw_action.get("action")
                or raw_action.get("tool_name")
                or raw_action.get("name")
                or ""
            ).strip()
            if not tool_name or tool_name not in tool_schema_map:
                continue

            function_def = tool_schema_map[tool_name]
            parameter_schema = function_def.get("parameters", {}) or {}
            properties = parameter_schema.get("properties", {}) or {}
            parameters = dict(raw_action.get("parameters") or {})

            if "conversation_id" in properties and "conversation_id" not in parameters:
                parameters["conversation_id"] = conversation_id

            normalized_actions.append(
                ActionSpec(
                    tool_name=tool_name,
                    parameters=parameters,
                    source="quick_reply",
                    metadata={
                        "conversation_id": conversation_id,
                        "follow_up_type": "tool_action",
                    },
                )
            )

        return normalized_actions

    def _build_follow_up_actions(
        self,
        conversation_id: str,
        decision: str,
        next_reason: str,
        turn_index: int,
        max_turns: int,
        extra_actions: List[ActionSpec],
    ) -> List[ActionSpec]:
        follow_up_actions = list(extra_actions)

        if decision == "quick_reply" and turn_index < max_turns:
            follow_up_actions.append(
                ActionSpec(
                    tool_name="quick_reply",
                    parameters={
                        "conversation_id": conversation_id,
                        "reason": next_reason or "继续进行一轮轻量回复。",
                        "turn_index": turn_index + 1,
                        "max_turns": max_turns,
                    },
                    source="quick_reply",
                    metadata={"conversation_id": conversation_id, "follow_up_type": "quick_reply"},
                )
            )
        elif decision == "deep_chat":
            follow_up_actions.append(
                ActionSpec(
                    tool_name="enter_deep_chat",
                    parameters={
                        "conversation_id": conversation_id,
                        "reason": next_reason or "当前话题需要转入更深入的会话处理。",
                    },
                    source="quick_reply",
                    metadata={"conversation_id": conversation_id, "follow_up_type": "deep_chat"},
                )
            )

        return follow_up_actions

    async def execute(
        self,
        conversation_id: str,
        reason: str,
        turn_index: int = 1,
        max_turns: int = 2,
        **kwargs,
    ) -> ToolResult:
        qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
        chat_stream = await qq_chat_data.get_or_create_stream_by_id(conversation_id)
        if not chat_stream:
            logger.error(f"找不到会话流: {conversation_id}")
            return ToolResult(
                success=False,
                summary=f"未找到会话 '{conversation_id}'，无法执行快速回复。",
                error_message="chat_stream_not_found",
            )

        conversation_name = chat_stream.conversation_info.conversation_name
        reply_summary = await self._replyer.execute(
            reason=reason,
            chat_stream=chat_stream,
        )
        result_summary = f"在会话“{conversation_name}”中完成了一次快速回复。{reply_summary}".strip()

        if turn_index >= max_turns:
            return ToolResult(success=True, summary=result_summary)

        latest_history = chat_stream.build_chat_history_has_msg_id()
        tool_schemas = self._get_follow_up_tool_schemas()
        tool_schema_map = self._build_tool_schema_map(tool_schemas)
        follow_up_prompt = self._build_follow_up_prompt(
            conversation_name=conversation_name,
            reason=reason,
            reply_summary=reply_summary,
            latest_history=latest_history,
            turn_index=turn_index,
            max_turns=max_turns,
            tool_schemas=tool_schemas,
        )

        decision = "exit"
        next_reason = ""
        extra_actions: List[ActionSpec] = []
        try:
            llm_request = self._llm_factory.get_request("planner")
            content, _ = await llm_request.execute(prompt=follow_up_prompt)
            decision_payload = self._parse_follow_up_decision(content)
            decision = decision_payload["decision"]
            next_reason = decision_payload["reason"]
            extra_actions = self._normalize_tool_actions(
                decision_payload["actions"],
                conversation_id=conversation_id,
                tool_schema_map=tool_schema_map,
            )
            logger.info(
                f"会话 {conversation_id} 的 quick_reply 后效决策: {decision} "
                f"(reason={next_reason}, extra_actions={len(extra_actions)})"
            )
        except Exception as exc:
            logger.warning(f"quick_reply 后效决策失败，默认退出: {exc}")

        follow_up_actions = self._build_follow_up_actions(
            conversation_id=conversation_id,
            decision=decision,
            next_reason=next_reason,
            turn_index=turn_index,
            max_turns=max_turns,
            extra_actions=extra_actions,
        )

        return ToolResult(
            success=True,
            summary=result_summary,
            follow_up_action=follow_up_actions,
        )
