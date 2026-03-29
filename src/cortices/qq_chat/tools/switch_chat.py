import json
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from sqlmodel import select

from src.common.database.database_model import ConversationInfoDB, EventDB
from src.common.di.container import container
from src.cortices.tools_base import BaseTool
from src.agent.world_model import WorldModel
from src.cortices.qq_chat.data_model.qq_chat_data import QQChatData
from src.common.database.database_manager import DatabaseManager
from src.llm_api.factory import LLMRequestFactory
from src.common.action_model.action_spec import ActionSpec
from src.common.action_model.tool_result import ToolResult
from src.common.logger import get_logger

logger = get_logger("qq_chat")

if TYPE_CHECKING:
    from src.cortices.manager import CortexManager

class SwitchQQChatTool(BaseTool):
    """
    跨会话切换工具：从当前聊天切换到另一个群聊或私聊。
    核心逻辑：它会携带“来自哪里”、“为什么切换”以及“前一会话的关键信息”，防止上下文断裂。
    """

    def __init__(self, cortex_manager: "CortexManager"):
        super().__init__(cortex_manager)
        self._world_model = container.resolve(WorldModel)
        self.database_manager = container.resolve(DatabaseManager)
        self.llm_request_factory = container.resolve(LLMRequestFactory)

    @property
    def scope(self) -> List[str]:
        # 主要用于 deep_chat 过程中，当发现需要去另一个群完成任务时调用
        return ["deep_chat", "qq_app"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "switch_qq_chat",
            "description": "从当前会话切换到另一个 QQ 群聊或私聊，并携带当前上下文意图。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_conversation_id": {
                        "type": "string",
                        "description": "当前（老）会话的 ID"
                    },
                    "switch_reason": {
                        "type": "string",
                        "description": "为什么要切换？例如：去群B核实群A提到的某个信息"
                    },
                    "context_summary": {
                        "type": "string",
                        "description": "老会话中与此次切换相关的关键信息摘要"
                    }
                }
            },
            "required": ["source_conversation_id", "switch_reason"]
        }

    async def _get_chat_list_with_preview(self) -> str:
        """模仿 EnterQQApp，但增加针对性，展示所有可选会话。"""
        qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
        all_conversations: List[ConversationInfoDB] = await self.database_manager.get_all(select(ConversationInfoDB))
        
        formatted_list = ["## 可切换的 QQ 会话列表:"]
        for conv in all_conversations:
            stream_id = str(conv.conversation_id)
            chat_stream = qq_chat_data.streams.get(stream_id) if qq_chat_data else None
            unread_count = chat_stream.unread_count if chat_stream else 0
            
            type_str = "群聊" if conv.conversation_type == "group" else "私聊"
            status = f"【{unread_count}条未读】" if unread_count > 0 else "【活跃】"
            formatted_list.append(f"- [{type_str}] {conv.conversation_name} (ID: {stream_id}) {status}")
            
        return "\n".join(formatted_list)

    def _build_switch_planner_prompt(self, source_id: str, reason: str, context_summary: str, chat_list: str) -> str:
        """构建专门用于‘跨群行动’的编排提示词。"""
        context = self._world_model.get_context_for_motive()
        available_tools = self.cortex_manager.get_tool_schemas(scopes=["qq_app"])
        
        prompt = f"""
你正在进行【跨会话行动】。
你刚刚在会话 {source_id} 中，因为以下原因决定切换会话：
> "{reason}"

## 之前会话的关键上下文：
{context_summary or "无具体摘要"}

## 可供选择的目标会话列表：
{chat_list}

## 任务：
请根据切换原因，选择一个或多个目标会话，并规划下一步行动。
注意：你要去目标会话中完成与“切换原因”相关的事情。

## 可用的 Action 选项：
{json.dumps(available_tools, ensure_ascii=False, indent=2)}

输出格式：
{{
  "actions": [
    {{ "action": "quick_reply", "parameters": {{ "conversation_id": "目标ID", "reason": "衔接上下文的回复原因" }} }}
  ]
}}
"""
        return prompt

    async def execute(self, source_conversation_id: str, switch_reason: str, context_summary: str = "", **kwargs) -> ToolResult:
        try:
            # 1. 获取全局会话列表（模仿 EnterQQApp）
            chat_list = await self._get_chat_list_with_preview()
            
            # 2. 调用 LLM 重新编排，此时 LLM 知道“来龙去脉”
            prompt = self._build_switch_planner_prompt(source_conversation_id, switch_reason, context_summary, chat_list)
            llm_request = self.llm_request_factory.get_request("planner")
            content, _ = await llm_request.execute(prompt=prompt)
            
            # 3. 解析并返回 ActionSpec
            res_text = content.strip()
            if "```" in res_text:
                res_text = res_text.split("```json")[-1].split("```")[0].strip()
            
            plan_data = json.loads(res_text)
            action_items = plan_data.get("actions", [])
            
            planned_specs: List[ActionSpec] = []
            for item in action_items:
                spec = ActionSpec(
                    tool_name=item.get("action"),
                    parameters=item.get("parameters", {}),
                    source=f"switch_from_{source_conversation_id}" # 标记来源
                )
                planned_specs.append(spec)

            return ToolResult(
                success=True,
                summary=f"已从 {source_conversation_id} 切换，意图：{switch_reason}",
                follow_up_action=planned_specs
            )

        except Exception as e:
            logger.error(f"SwitchQQChat 失败: {e}", exc_info=True)
            return ToolResult(success=False, summary="跨会话切换失败", error_message=str(e))