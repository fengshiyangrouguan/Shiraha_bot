import json
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from src.cortices.tools_base import BaseTool
from src.agent.world_model import WorldModel
from src.cortices.reading.reading_data import ReadingData,Book
from datetime import datetime
from src.llm_api.factory import LLMRequestFactory
from src.common.action_model.action_spec import ActionSpec
from src.common.action_model.tool_result import ToolResult
from src.common.logger import get_logger

if TYPE_CHECKING:
    from src.cortices.manager import CortexManager

logger = get_logger("reading")

class EnterLibraryTool(BaseTool):
    """
    模拟“打开书库/书架”的动作。
    它会展示当前所有书籍的阅读状态和进度，并由决策器决定下一步：
    是继续阅读、切换书籍，还是对新书进行初步浏览。
    """

    def __init__(self, world_model: WorldModel, cortex_manager: "CortexManager", llm_request_factory: "LLMRequestFactory"):
        super().__init__(cortex_manager)
        self.world_model = world_model
        self.llm_request_factory = llm_request_factory

    @property
    def scope(self) -> str:
        return ["main"] # 放置在主 Planner 的工具箱中

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "enter_library",
            "description": "进入书房，查看阅读进度并决定下一步阅读计划。",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "进入书房的目的"
                    }
                }
            },
            "required": ["objective"]
        }

    async def get_domain_summary(self) -> str:
        """
        获取书房皮层的实时感知摘要，直接展示书架书籍列表和阅读进度。
        """
        reading_data: ReadingData = await self._world_model.get_cortex_data("reading_data")
        
        # 1. 基础校验
        if not reading_data or not reading_data.book_dict:
            return "### 书房状态\n当前书架为空，无书籍记录。"

        summary_lines = ["### 书房状态概览"]

        # 2. 检查是否有新上架通知（来自扫描任务）
        if "书房" in self._world_model.notifications:
            summary_lines.append(f"通知: {self._world_model.notifications['书房']}")

        # 3. 遍历书架详细信息 (同步 EnterLibraryTool 的逻辑)
        summary_lines.append("--- 书架列表 ---")
        for book in reading_data.book_dict.values():
            # 进度计算逻辑优化：防止除以 0，确保 total_chunks 已被初始化
            total = book.total_chunks if hasattr(book, 'total_chunks') else 0
            current = book.current_chunk_index if hasattr(book, 'current_chunk_index') else 0
            
            progress = (current / total * 100) if total > 0 else 0.0
            
            status_tag = book.status
            if getattr(book, 'is_finished_reading', False):
                status_tag = "已读完"

            book_info = f"- 《{book.book_title}》 [{status_tag}] | 进度: {progress:.1f}%"
            
            # 只有读过的书才显示时间
            if status_tag != "新书未读" and getattr(book, 'last_read_time', None):
                from datetime import datetime
                dt = datetime.fromtimestamp(book.last_read_time)
                book_info += f" | 上次阅读: {dt.strftime('%Y-%m-%d %H:%M')}"
            
            summary_lines.append(book_info)

        # 4. 当前正打开的书籍
        if reading_data.current_reading_book:
            summary_lines.append(f"\n当前案头正翻开的书: 《{reading_data.current_reading_book.book_title}》")

        return "\n".join(summary_lines)

    async def _run_decide_planner(self, objective: str, library_str: str) -> Dict[str, Any]:
        """内置轻量决策器"""
        context = self.world_model.get_context_for_motive()
        available_tools = self.cortex_manager.get_tool_schemas(scopes=["reading"]) # 只获取阅读相关的子工具
        short_term_memory = "以下是按时间顺序排列的近期活动：\n"+"\n".join(self.world_model.short_term_memory)
        
        prompt = f"""
你叫 {context['bot_name']}
你是{context['bot_identity']}
你的性格是 {context['bot_personality']}，
你的兴趣包括 {context['bot_interest']}。

现在是 {context['time']}。
此刻你的心理状态是：{context['mood']}。

你的近期活动：
{short_term_memory}

你现在站在书架前。你现在的意图是："{objective}"

你的书架：
{library_str}

## 决策规则
1. 如果某本书已经读完，除非是为了复习，否则应优先寻找新书。
2. 考虑你的心理状态：如果疲劳度高，建议选择短小的片段或退出；如果求知欲强，选择与目标相关的书籍。
3. 如果书架上没有想读的书，请选择退出。

可用动作：
{json.dumps(available_tools, ensure_ascii=False, indent=2)}

## 输出格式 (JSON):
{{
  "decision": "tool_call",
  "tool_name": "<工具名称>",
  "parameters": {{...}},
  "thought": "<你做出这个决定的思考过程>"
}}
或者
{{
  "decision": "exit",
  "thought": "<原因>"
}}
"""
        llm_request = self.llm_request_factory.get_request("planner")
        content, _ = await llm_request.execute(prompt=prompt)
        try:
            return json.loads(s=content.strip().replace("```json", "").replace("```", ""))
        except:
            return {"decision": "exit", "thought": "我思维突然断片了，打算先离开书架。"}

    async def execute(self, objective: str) -> ToolResult:
        # 先清空新书上架通知
        self.world_model.notifications.pop("书房", None)
        library_str = await self._get_library_context()
        decision = await self._run_decide_planner(objective, library_str)
        
        if decision.get("decision") == "tool_call":
            thought = decision.get("thought")
            logger.info(thought)
            tool_name = decision.get("tool_name")
            params = decision.get("parameters", {})
            # 执行具体的阅读动作（如 start_read_chunk）
            try:
                action = ActionSpec(tool_name=tool_name, parameters=params)
                result = await self.cortex_manager.execute_action(action)
                return ToolResult(
                    success=result.success,
                    summary=result.summary,
                    error_message=result.error_message,
                    follow_up_action=result.follow_up_action,
                )
            # TODO:也改为形如qqcortex的agentloop为中心的调用模式
            except Exception as e:
                return ToolResult(
                    success=False,
                    summary="在执行 '{tool_name}' 时出错",
                    error_message= e
                )

        logger.info(f"决定离开书架。原因是{decision.get('thought', '无')}")
        return ToolResult(
            success=True,
            summary=f"决定离开书架。原因是{decision.get('thought', '无')}",
        )
