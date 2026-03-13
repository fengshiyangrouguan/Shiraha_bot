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
from src.common.logger import get_logger

logger = get_logger("qq_chat")

if TYPE_CHECKING:
    from src.cortices.manager import CortexManager


class EnterQQAppTool(BaseTool):
    """
    该工具模拟用户“打开QQ”的动作。
    它会首先获取所有会话的概览（包括未读消息情况），然后结合来自更高层级的意图，
    通过一个内置的“决策器”来智能判断下一步行动。
    它现在可以直接处理简单的工具调用（如发送消息），或将复杂任务（如批量处理）委派出去。
    """

    def __init__(self, cortex_manager: "CortexManager"):
        super().__init__(cortex_manager)
        self._world_model = container.resolve(WorldModel)
        self.database_manager = container.resolve(DatabaseManager)
        self.llm_request_factory = container.resolve(LLMRequestFactory)

    @property
    def scope(self) -> str:
        return "main"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "enter_qq_app",
            "description": "启动QQ应用进行决策分诊，可直接处理简单回复或委派复杂任务。",
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
        """Formats a list of EventDB messages into readable strings."""
        formatted_messages = []
        for msg in messages:
            sender_cardname = msg.user_cardname or "未知用户"
            sender_nickname = msg.user_nickname or "未知用户"
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.time))
            content = msg.event_content or "[消息内容为空]"
            line = f"    - [{timestamp_str}] {sender_nickname}: {content}"
            formatted_messages.append(line)
        return formatted_messages

    async def _get_conversation_info(self) -> tuple[Optional[str], Optional[QQChatData]]:
        """获取会话列表，并格式化为文本, 同时在下方插入未读消息（最多15条）。"""
        qq_chat_data: QQChatData = await self.world_model.get_cortex_data("qq_chat_data")
        all_conversations: List[ConversationInfoDB] = await self.database_manager.get_all(select(ConversationInfoDB))
        
        formatted_list = ["[会话列表]"]
        
        for conv in all_conversations:
            stream_id = f"{conv.conversation_id}"
            unread_count = 0
            chat_stream = qq_chat_data.streams.get(stream_id) if qq_chat_data else None
            if chat_stream:
                unread_count = chat_stream.unread_count
            
            type_str = "群聊" if conv.conversation_type == "group" else "私聊"
            messages_to_fetch = 0
            if unread_count > 0:
                messages_to_fetch = min(unread_count, 15)
                unread_str = f"，有 {unread_count} 条未读消息" + (f" (仅显示最新的{messages_to_fetch}条)" if unread_count > 15 else "")
            else:
                unread_str = "，无未读消息"

            formatted_list.append(f"- [{type_str}] {conv.conversation_name} (ID: {stream_id}){unread_str}")

            if messages_to_fetch > 0 and chat_stream:
                query = (select(EventDB).where(EventDB.conversation_id == chat_stream.conversation_info.conversation_id).order_by(EventDB.time.desc()).limit(messages_to_fetch))
                messages_desc: List[EventDB] = await self.database_manager.get_all(query)
                messages = list(reversed(messages_desc))
                formatted_messages = self._format_messages(messages)
                formatted_list.extend(formatted_messages)
                chat_stream.mark_as_read()
            
        await self.world_model.save_cortex_data("qq_chat_data", qq_chat_data)
        return "\n".join(formatted_list), qq_chat_data
    

    async def _run_decide_planner(self, objective: str, context_str: str) -> Dict[str, Any]:
        """运行轻量判断器（LLM调用）来决定下一步行动"""
        # 模仿 main_planner, 获取可用工具
        context = self.world_model.get_context_for_motive()
        available_tools = self.cortex_manager.get_tool_schemas(scope="qq_app")
        time = self.world_model.get_current_time_string()
        short_term_memory = "以下是按时间顺序排列的近期活动记忆：\n"+"\n".join(self.world_model.short_term_memory)
# 你的兴趣包括 {context['bot_interest']}。
        prompt = f"""
你叫 {context['bot_name']}。
你是 {context['bot_identity']}。
你的性格是 {context['bot_personality']}，

{time}
此刻你的心理状态是：{context['mood']}。

你的近期活动记忆：
{short_term_memory}
        
现在你正在qq软件中浏览会话列表，正在规划下一步行动。
你当前的总意图是："{objective}"

{context_str}

## **社交规范注意**
1.**禁止强行切入**：不要因为对方聊到了你的【兴趣爱好】就生硬地发起话题或过度讨论该关键词。
2.**社交克制**：不要为了展示你的“兴趣爱好”而生硬地开启话题。不要刻意突出自身兴趣爱好，自然的融入聊天。
3.**自然流露**：兴趣应该是你性格的底色，而不是你说话的模板。只有在氛围合适时才轻量提及，不要像复读兴趣标签的机器。
4.**保持随性**：你没有义务回应每一句未读消息。如果话题不投机/不感兴趣/和你没有关系，你可以决定旁观不回复，或者也可以按照你的性格认知下的社交礼仪去决定怎么做。
6.不要对表情包进行回复


## 决策规则
根据你的意图和上面的会话列表，从以下选项中选择一个最合适的行动，并严格按照JSON格式输出。

你可以选择以下决策行动：

## **直接退出 (`exit`)**: 如果因某种原因不想聊天或无法聊天。    
## 输出格式：
```json
{{
  "decision": "exit",
  "reason": "<说明退出的原因>"
}}
```

其他可用行动列表：
```json
{json.dumps(available_tools, ensure_ascii=False, indent=2)}
```

## **输出格式:**
```json
{{
  "decision": "tool_call",
  "tool_name": "<工具名称>",
  "parameters": {{...}}
}}
```
---
请严格按照以上JSON格式之一输出你的决策。
"""     
        llm_factory = self.llm_request_factory
        llm_request = llm_factory.get_request("planner")

        content, model_name = await llm_request.execute(prompt=prompt)
        response = content.strip()
        try:
            json_str = response.strip().replace("```json", "").replace("```", "")
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"decision": "exit", "reason": "无法解析决策器的输出。"}


 
    async def execute(self, objective: str) -> str:
        """主执行函数：收集信息 -> 决策 -> 执行或委派"""
        context_str, qq_chat_data = await self._get_conversation_info()
        if not qq_chat_data:
            return "QQ未连接网络，先退出QQ了。"

        decision = await self._run_decide_planner(objective, context_str)
        decision_type = decision.get("decision")
        logger.info(f"原始决策：{decision}")

        if decision_type == "tool_call":
            tool_name = decision.get("tool_name")
            parameters = decision.get("parameters", {})
            if not tool_name:
                return "行动无效：QQ卡了"
            
            try:
                # 直接执行工具调用
                result = await self.cortex_manager.call_tool_by_name(tool_name, **parameters)
                return f"{result}"
            except Exception as e:
                return f"在执行 '{tool_name}' 时出错: {e}"
        elif decision_type == "exit":
            if decision.get("reason"):
                reason = decision.get("reason")
                return reason
        else:
            return "不知道该干什么，退出QQ应用了。"
