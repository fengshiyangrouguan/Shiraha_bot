# shirahabot/planner.py
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.plugin_system.core.plugin_loader import PluginLoader
from src.plugin_system.core.plugin_manager import PluginManager
from src.utils.logger import logger
from src.llm_api.request import LLMRequest
from src.config import LLM_MODELS

SYSTEM_PROMPT = """
你是一个智能任务规划器。

你的目标：
1. 理解用户意图。
2. 判断是否需要调用可用工具中的工具。
3. 如果需要调用工具，严格返回以下 JSON：

{
  "tool_calls": [
    {
      "name": "工具名称",
      "arguments": { 参数字典 }
    }
  ]
}

如果不需要调用工具，返回：
{}

规则：
- 不要输出解释文字。
- 不要输出注释。
- 不要输出与 JSON 无关的任何内容。
- 参数必须是合法 JSON，可被 json.loads() 成功解析。
"""


class Planner:
    """
    Planner（调度层）

    职责：
    - 向 LLM 提供 Tool 声明
    - 解析 LLM 决策
    - 委托 PluginManager 创建 Tool 实例
    - 执行 Tool
    """

    def __init__(self, plugin_manager: PluginManager, plugin_loader: PluginLoader | None = None):
        self.plugin_manager = plugin_manager
        self.plugin_loader = plugin_loader
        self.llm_request = LLMRequest(model_configs=LLM_MODELS)

    def initialize_plugins(self, plugin_root: str | Path | None = None) -> None:
        """使用当前版本的 Loader + Manager 初始化插件。"""
        if plugin_root is not None:
            self.plugin_loader = PluginLoader(Path(plugin_root))

        if self.plugin_loader is None:
            raise ValueError("plugin_loader 未设置，无法初始化插件")

        plugin_infos = self.plugin_loader.load_plugins()
        self.plugin_manager.initialize_from_infos(plugin_infos)
        logger.info(f"Planner 已初始化插件系统，共加载声明 {len(plugin_infos)} 个")

    async def plan_and_execute(
            self,
            user_prompt: str,
            chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        logger.info("Planner 开始工作")

        # 1、从 Manager 获取声明态 tool 定义（给 LLM）
        try:
            tools = self.plugin_manager.get_all_tool_definitions()
        except Exception as e:
            logger.error(f"获取工具定义失败: {e}")
            return ""

        if not tools:
            logger.info("当前没有可用工具")
            return ""

        # 2、调用 LLM 做规划
        llm_response = await self._call_llm(user_prompt, tools, chat_history)
        tool_calls = llm_response.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            logger.warning(f"LLM 返回的 tool_calls 不是列表: {type(tool_calls).__name__}")
            return ""

        if not tool_calls:
            logger.info("LLM 决定不调用任何工具")
            return ""

        logger.info(f"LLM 决定调用工具: {[c.get('name') for c in tool_calls if isinstance(c, dict)]}")

        # 3、 执行工具
        summaries: List[str] = []

        for call in tool_calls:
            if not isinstance(call, dict):
                logger.warning(f"忽略非法 tool_call 项: {call}")
                continue

            tool_name = call.get("name")
            arguments = call.get("arguments", {})
            if not isinstance(tool_name, str) or not tool_name.strip():
                logger.warning(f"忽略缺少 name 的 tool_call: {call}")
                continue
            if not isinstance(arguments, dict):
                logger.warning(f"工具 '{tool_name}' 参数不是对象，已忽略")
                continue

            tool_instance = self.plugin_manager.create_tool_instance(tool_name)
            if not tool_instance:
                msg = f"你调用了 {tool_name} 工具，但该工具不存在。"
                logger.warning(msg)
                summaries.append(msg)
                continue

            try:
                logger.info(f"执行工具 '{tool_name}'，参数: {arguments}")
                output = await tool_instance.execute(**arguments)

                if "error" in output:
                    msg = f"你调用了 {tool_name} 工具，执行失败：{output['error']}"
                    logger.error(msg)
                elif "result" in output:
                    msg = f"你调用了 {tool_name} 工具，得到结果：{output['result']}"
                    logger.info(f"工具 '{tool_name}' 执行成功")
                else:
                    msg = f"你调用了 {tool_name} 工具，但返回格式不合法"
                    logger.warning(f"工具 '{tool_name}' 返回异常格式: {output}")

                summaries.append(msg)

            except Exception as e:
                msg = f"你调用了 {tool_name} 工具，执行异常：{e}"
                logger.exception(msg)
                summaries.append(msg)

        return "\n".join(summaries)

    async def _call_llm(
            self,
            user_prompt: str,
            tools: List[Dict[str, Any]],
            chat_history: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        调用 LLM 生成 tool_calls JSON
        """
        tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
        history_json = json.dumps(self._normalize_chat_history(chat_history), ensure_ascii=False)

        full_prompt = f"""
{SYSTEM_PROMPT}

用户输入:
{user_prompt}

历史对话（可为空）:
{history_json}

可用工具:
{tools_json}

请输出 JSON：
"""

        response = await self.llm_request.generate_response_async(full_prompt)

        response_str = response[0] if isinstance(response, tuple) else response
        logger.debug(f"LLM 原始返回: {response_str}")

        try:
            return json.loads(response_str)
        except Exception:
            logger.error(f"LLM 返回不是合法 JSON，忽略工具调用: {response_str}")
            return {}

    @staticmethod
    def _normalize_chat_history(chat_history: Optional[List[Any]]) -> List[Dict[str, Any]]:
        """将聊天历史规范化为可 JSON 序列化的数据结构。"""
        normalized: List[Dict[str, Any]] = []
        if not chat_history:
            return normalized

        for item in chat_history:
            if isinstance(item, dict):
                normalized.append(item)
                continue

            normalized.append(
                {
                    "user_id": getattr(item, "user_id", ""),
                    "content": getattr(item, "content", str(item)),
                    "timestamp": getattr(item, "timestamp", None),
                }
            )

        return normalized
