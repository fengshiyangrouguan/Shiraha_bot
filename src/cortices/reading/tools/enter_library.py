import json
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from src.cortices.tools_base import BaseTool
from src.agent.world_model import WorldModel
from src.cortices.reading.reading_data import ReadingData,Book
from datetime import datetime
from src.llm_api.factory import LLMRequestFactory
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
        return "main" # 放置在主 Planner 的工具箱中

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

    async def _get_library_context(self) -> str:
        """获取书架现状的文本描述"""
        reading_data: ReadingData = await self.world_model.get_cortex_data("reading_data")
        if not reading_data or not reading_data.book_dict.values():
            return "书架是空的欸，没有找到任何书。你该去找别人推荐点书单了"

        lines = ["[书架状态概要]"]
        book:Book = None
        for book in reading_data.book_dict.values():
            progress = (book.current_chunk_index / book.total_chunks * 100) if book.total_chunks > 0 else 0
            status_tag = book.status
            if book.is_finished_reading:
                status_tag = "已读完"

            book_str = f"- {book.book_title} [{status_tag} ] | 进度: {progress:.1f}%"   
            if book.status != "新书未读":
                timestamp = book.last_read_time
                dt_object = datetime.fromtimestamp(timestamp)
                time = dt_object.strftime("%Y-%m-%d %H:%M")
                book_str = book_str + f"，上次阅读时间：{time}"
            lines.append(book_str)
        
        if reading_data.current_reading_book:
            lines.append(f"\n当前案头正翻开的书: 《{reading_data.current_reading_book.book_title}》")
            
        return "\n".join(lines)

    async def _run_decide_planner(self, objective: str, library_str: str) -> Dict[str, Any]:
        """内置轻量决策器"""
        context = self.world_model.get_context_for_motive()
        available_tools = self.cortex_manager.get_tool_schemas(scope="reading") # 只获取阅读相关的子工具
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
  "reason": "<原因>"
}}
"""
        llm_request = self.llm_request_factory.get_request("planner")
        content, _ = await llm_request.execute(prompt=prompt)
        try:
            return json.loads(s=content.strip().replace("```json", "").replace("```", ""))
        except:
            return {"decision": "exit", "reason": "我思维突然断片了，打算先离开书架。"}

    async def execute(self, objective: str) -> str:
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
                result = await self.cortex_manager.call_tool_by_name(tool_name, **params)
                return f"{result}"
            except Exception as e:
                return f"在执行 '{tool_name}' 时出错: {e}"

        logger.info(f"决定离开书架。原因是{decision.get('reason', '无')}")
        return f"{decision.get('reason', '无')}"