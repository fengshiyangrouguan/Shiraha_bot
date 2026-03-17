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


class EnterQQAppTool(BaseTool):
    """
    编排型工具：模拟打开QQ，查看全局会话状态，并规划一系列后续行动指令。
    该工具不再直接执行终端动作，而是返回 ActionSpec 列表由外层统一编排。
    """

    def __init__(self, cortex_manager: "CortexManager"):
        super().__init__(cortex_manager)
        self._world_model = container.resolve(WorldModel)
        self.database_manager = container.resolve(DatabaseManager)
        self.llm_request_factory = container.resolve(LLMRequestFactory)

    @property
    def scope(self) -> List[str]:
        # 允许在 main 和 deep_chat 作用域下被调用
        return ["main", "deep_chat"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "enter_qq_app",
            "description": "打开/回到QQ的主页面，查看全局会话列表和未读消息并规划后续行动。",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "总体意图，详细说明这次打开qq是为了什么"
                    }
                }
            },
            "required": ["objective"]
        }

    def _format_messages(self, messages: List[EventDB]) -> List[str]:
        """将数据库消息格式化为可读字符串。"""
        formatted_messages = []
        for msg in messages:
            sender_nickname = msg.user_nickname or "未知用户"
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(msg.time))
            content = msg.event_content or "[内容为空]"
            formatted_messages.append(f"    - [{timestamp_str}] {sender_nickname}: {content}")
        return formatted_messages

    async def _get_global_context_str(self) -> str:
        """获取所有会话的未读状态概览。"""
        qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data")
        all_conversations: List[ConversationInfoDB] = await self.database_manager.get_all(select(ConversationInfoDB))
        
        formatted_list = ["## 当前QQ会话列表概览:"]
        
        for conv in all_conversations:
            stream_id = str(conv.conversation_id)
            chat_stream = qq_chat_data.streams.get(stream_id) if qq_chat_data else None
            unread_count = chat_stream.unread_count if chat_stream else 0
            
            type_str = "群聊" if conv.conversation_type == "group" else "私聊"
            unread_info = f"【{unread_count}条未读】" if unread_count > 0 else "【已读】"
            
            formatted_list.append(f"- [{type_str}] {conv.conversation_name} (ID: {stream_id}) {unread_info}")

            # 如果有未读消息，提取最近几条辅助决策
            if unread_count > 0 and chat_stream:
                query = (select(EventDB)
                         .where(EventDB.conversation_id == chat_stream.conversation_info.conversation_id)
                         .order_by(EventDB.time.desc())
                         .limit(min(unread_count, 5))) # 仅查看最近5条辅助规划
                messages_desc = await self.database_manager.get_all(query)
                formatted_list.extend(self._format_messages(list(reversed(messages_desc))))
        return "\n".join(formatted_list)

    def _build_planner_prompt(self, objective: str, context_str: str) -> str:
        """构建中文编排提示词。"""
        context = self._world_model.get_context_for_motive()
        # 获取 qq_app 作用域下的子工具（如 batch_quick_plan, quick_reply 等）
        available_tools = self.cortex_manager.get_tool_schemas(scopes=["qq_app"])
        # 过滤掉自身
        available_tools = [t for t in available_tools if t.get("function", {}).get("name") != "enter_qq_app"]
        
        time_str = self._world_model.get_current_time_string()
        memory_str = "\n".join(self._world_model.short_term_memory)

        prompt = f"""
你叫 {context['bot_name']}，身份是 {context['bot_identity']}。
性格设定：{context['bot_personality']}
当前情绪：{context['mood']}
当前时间：{time_str}

## 近期记忆：
{memory_str}

## 当前任务目标：
"{objective}"

## QQ全局实时状态：
{context_str}

## 社交规划准则：
1. **需要外部支持**：如果回复需要外部事实或工具结果的支持，你应该先调用相关工具，最后再执行回复。
2. **随性自然**：如果对聊天不感兴趣或与你无关，可以直接 `exit`。不要像机器人一样强迫回复。
3. **多步规划**：你可以一次性规划多个群聊的多个动作。按照你自己的想法来
例1：先调用某个工具查询信息，再进行回复。 例2：在多个群同时规划reply挨个回复消息

## 可用的 Action 选项：
- **exit**: 退出QQ，不进行任何操作。参数：{{"reason": "理由"}}
- **其他可用工具**:
{json.dumps(available_tools, ensure_ascii=False, indent=2)}

## 输出格式（严格 JSON）：
{{
  "actions": [
    {{
      "action": "工具名",
      "parameters": {{ ... }}
    }}
  ]
}}
"""
        return prompt

    async def execute(self, objective: str, **kwargs) -> ToolResult:
        """主执行函数：分析全局 -> 批量编排 -> 返回动作序列"""
        try:
            # 1. 采集全局上下文
            context_str = await self._get_global_context_str()
            
            # 2. 调用 LLM 进行多步规划
            prompt = self._build_planner_prompt(objective, context_str)
            llm_request = self.llm_request_factory.get_request("planner")
            content, _ = await llm_request.execute(prompt=prompt)
            
            # 3. 解析编排结果
            res_text = content.strip()
            if "```" in res_text:
                res_text = res_text.split("```json")[-1].split("```")[0].strip()
            
            plan_data = json.loads(res_text)
            action_items = plan_data.get("actions", [])
            
            if not action_items:
                return ToolResult(success=True, summary="没有规划任何行动，直接退出。")

            planned_specs: List[ActionSpec] = []
            summary_parts = []

            for item in action_items:
                name = item.get("action")
                params = item.get("parameters", {})
                
                if name == "exit":
                    summary_parts.append(f"决定退出: {params.get('reason', '无')}")
                    continue
                
                # 构造 ActionSpec
                spec = ActionSpec(
                    tool_name=name,
                    parameters=params,
                    source="enter_qq_app"
                )
                planned_specs.append(spec)
                summary_parts.append(f"规划了动作: {name}")

            final_summary = " | ".join(summary_parts) if summary_parts else "规划完成"
            
            return ToolResult(
                success=True,
                summary=final_summary,
                follow_up_action=planned_specs
            )

        except Exception as e:
            logger.error(f"EnterQQApp 编排失败: {e}", exc_info=True)
            return ToolResult(success=False, summary="QQ应用编排逻辑出错", error_message=str(e))