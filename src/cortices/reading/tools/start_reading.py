import os
import json
from typing import Dict, Any, Optional, TYPE_CHECKING
from src.cortices.tools_base import BaseTool
from src.agent.world_model import WorldModel
from src.common.action_model.action_spec import ActionSpec
from src.common.action_model.tool_result import ToolResult
from src.cortices.reading.reading_data import ReadingData, Book
from src.llm_api.factory import LLMRequestFactory
from src.common.database.database_manager import DatabaseManager
from src.common.logger import get_logger
from collections import deque

logger = get_logger("reading")

if TYPE_CHECKING:
    from src.cortices.manager import CortexManager

class StratReadingTool(BaseTool):
    """
    子工具：拿起并开始阅读。
    功能：根据书名定位书籍，从本地文件加载切片内容，并设置为当前阅读对象。
    """

    def __init__(self, world_model: WorldModel, cortex_manager: "CortexManager",database_manager:"DatabaseManager",llm_request_factory: "LLMRequestFactory"):
        super().__init__(cortex_manager)
        self.world_model = world_model
        self.db_manager = database_manager
        self.llm_request_factory = llm_request_factory

    @property
    def scope(self) -> str:
        # 该工具仅在进入 Library 后可见
        return "reading"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "start_reading",
            "description": "从书架中选定一本书籍开始阅读。如果之前读过，将从上次位置续读。",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_title": {
                        "type": "string",
                        "description": "书籍的完整名字（必须与书架上注册的名字完全匹配）"
                    },
                    "initial_motive": {
                        "type": "string",
                        "description": "选择读这本书的动机"
                    }
                }
            },
            "required": ["book_title","initial_motive"]
        }
    async def execute(self, book_title: str, initial_motive: str) -> str:
        # 1. 初始化数据
        reading_data: ReadingData = await self.world_model.get_cortex_data("reading_data")
        target_book = reading_data.book_dict.get(book_title)
        
        if not target_book:
            return f"我在书架上找不到《{book_title}》。"

        # 2. 延迟加载 Chunks
        if not target_book.chunks:
            with open(target_book.registered_file_path, "r", encoding="utf-8") as f:
                target_book.chunks = f.read().split("\n\n---(segment)---\n\n")
        
        reading_data.current_reading_book = target_book
        initial_entry = f"你开始了《{book_title}》的阅读。初衷：{initial_motive}"
        reading_history = [initial_entry]
        reading_history_deque = deque([initial_entry], maxlen=3)

        # 3. 开启沉浸式循环
        motive = initial_motive
        while True:
            reading_data.set_book_status("在读")
            current_chunk = target_book.get_current_chunk()
            if not current_chunk:
                final_summary = await self.generate_reading_summary(book_title, reading_history)
                logger.info(f"我完全读完了《{book_title}》，总结：{final_summary}")
                await reading_data.update_book_progress_to_db(self.db_manager)
                return f"我已经把《{book_title}》全部读完了。总结：{final_summary}"

            # 运行决策器生成感想并决定下一步
            decision = await self._run_reading_planner(target_book, current_chunk, reading_history_deque, motive)
            
            # 记录感想和总结
            reflection_entry = (
                f"\n--- 第 {target_book.current_chunk_index + 1} 段笔记 ---\n"
                f"总结：{decision.get('summary')}\n"
                f"感悟：{decision.get('reflection')}\n"
            )
            logger.info(reflection_entry)
            reading_history.append(reflection_entry)
            reading_history_deque.append(reflection_entry)

            # 将感想存入短期记忆，影响白羽后续的动机
            self.world_model.add_flow_cache(f"我正在阅读《{book_title}》，{decision.get('summary')}，{decision.get('reflection')}")
            await reading_data.update_book_progress_to_db(self.db_manager)
            # 4. 判断下一步行动
            action = decision.get("action")
            if action == "continue":
                target_book.advance_to_next_chunk()
                # 实时保存进度
                await self.world_model.save_cortex_data("reading_data", reading_data)
                motive = decision.get("next_motive")
            elif action == "exit":
                await self.world_model.save_cortex_data("reading_data", reading_data)
                await reading_data.update_book_progress_to_db(self.db_manager)
                final_summary = await self.generate_reading_summary(book_title, reading_history)
                logger.info(f"我结束了阅读《{book_title}》，因为{decision.get('next_motive')},{final_summary}")
                return f"我决定放下书本。因为{decision.get('next_motive')}，阅读总结：{final_summary}"

    async def _run_reading_planner(self, book: Book, chunk: str, history: list,motive: str) -> Dict[str, Any]:
        """内部决策器：专注于阅读理解和动机转化"""
        context = self.world_model.get_context_for_motive()
        prompt = f"""
你叫 {context['bot_name']}，
你是 {context['bot_identity']}。
你的性格是 {context['bot_personality']}，
你的兴趣包括 {context['bot_interest']}。

现在是 {context['time']}。
当前心情：{context['mood']}

{context["notifications"]}
{context['alert']}

你正在阅读《{book.book_title}》的第 {book.current_chunk_index + 1}/{book.total_chunks} 段。
你的当前动机：{motive}


## 正在读的内容
{chunk}

## 之前的阅读记忆：
{''.join(history)}

## 指令
请基于上述内容完成一份读书笔记。你的笔记必须包含以下三个维度：
1. **summary**: 用你自己的语气，第一人称，以"我……"开头，简明扼要地复述这段内容的核心信息，要像是在日记里对自己说话。
2. **reflection**: 这段内容是否触动了你？以第一人称，写下你的感悟。
3. **next_motive**: 读完这段后，你想去做什么？请给出一个明确的动机(例如继续读/停止阅读去分享/换本书……)。
4. **action**: 决定 "continue" (继续读下一段) 或 "exit" (停止阅读)。

## 注意：
请不要忘记通知的情况
**连续阅读3段左右时请选择退出**


## 输出格式为严格 JSON：
{{
  "summary": "...",
  "reflection": "...",
  "next_motive": "...",
  "action": "continue/exit",
}}
"""
        llm_request = self.llm_request_factory.get_request("planner")
        content, _ = await llm_request.execute(prompt=prompt)
        # 清洗 JSON 格式并返回
        return json.loads(content.strip().replace("```json", "").replace("```", ""))
    
    async def generate_reading_summary(self, book_title: str, history: list) -> str:
        """
        使用小模型对本次阅读会话的所有感想进行最终复盘。
        """
        # 提取所有笔记中的感想部分，过滤掉系统提示
        reflections = "\n".join(history[1:])
        # 2. 简单的截断逻辑（防止超过 2048 Token）
    # 如果笔记太多，只取最近的 15 条笔记进行总结
        MAX_NOTES = 15
        raw_reflections = reflections
        if len(reflections) > MAX_NOTES:
            raw_reflections = raw_reflections[-MAX_NOTES:]
    
        reflections_text = "\n".join(raw_reflections)
        
        prompt = f"""
    你是一个总结器：
    你刚刚读完了《{book_title}》的几个片段。
    以下是你刚才随手记下的读书笔记：
    {reflections_text}

    请将这些零散的笔记整合，进行200字左右的压缩总结。
    要求：
    1. 保持第一人称，以"我……"开头
    2. 保留笔记中的行文风格
    3. 语气要自然，只对原文进行压缩精简，保留关键词，不添加任何原文没有的新的内容/词汇/感想，不换行
    """
        # 使用较小的 planner 或专用的 summary 模型
        llm_request = self.llm_request_factory.get_request("planner")
        summary, _ = await llm_request.execute(prompt=prompt)
        return summary.strip()