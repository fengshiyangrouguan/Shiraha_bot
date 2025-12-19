# E:\project\Shiraha_bot\shirahabot\planner.py
import json
from typing import List, Dict, Any, Optional

from src.plugin_system.core.plugin_manager import PluginManager
from src.utils.logger import logger
from src.llm_api.request import LLMRequest
from src.config import LLM_MODELS

DO_NOT_REPLY = object()

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

如果工具需要参数，请提取所需的参数填入"arguments"字段中。

规则：
- 不要输出解释文字。
- 不要输出注释。
- 不要输出与 JSON 无关的任何内容。
- 参数必须是合法 JSON，可被 json.loads() 成功解析。

下面开始任务。
"""

class Planner:
    """
    调用 LLM 根据用户输入自动规划并决定调用哪些工具。
    """
    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
        self.llm_request = LLMRequest(model_configs=LLM_MODELS)

    async def plan_and_execute(
        self,
        user_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        logger.info("规划器开始工作...")

        tools = self.plugin_manager.get_all_tool_definitions()
        if not tools:
            logger.info("没有可用工具，跳过工具规划。")
            return ''

        # 调用 LLM 决策
        llm_response = await self._call_llm(user_prompt, tools)
        tool_calls = llm_response.get("tool_calls")
        if not tool_calls:
            logger.info("LLM决定不调用任何工具。")
            return ''

        logger.info(f"LLM决定使用工具: {[call['name'] for call in tool_calls]}")

        # 拼接所有工具调用结果
        tool_summary_lines = []

        for call in tool_calls:
            tool_name = call.get("name")
            args = call.get("arguments", {})

            tool_class = self.plugin_manager.get_tool(tool_name)
            if not tool_class:
                msg = f"你调用了 {tool_name} 工具，但该工具不存在。"
                tool_summary_lines.append(msg)
                logger.warning(msg)
                continue

            try:
                logger.info(f"执行工具 '{tool_name}' 参数: {args}")
                instance = tool_class(self.plugin_manager.tool_to_plugin_map[tool_name].config)
                plugin_output = await instance.execute(**args)
                if "error" in plugin_output:
                    msg = f"你调用了 {tool_name} 工具，执行失败：{plugin_output['error']}"
                    logger.error(msg)
                elif "result" in plugin_output:
                    msg = f"你调用了 {tool_name} 工具，了解到：{plugin_output['result']}"
                    logger.info(f"工具 '{tool_name}' 执行成功，结果已解析。")
                else:
                    msg = f"你调用了 {tool_name} 工具，但返回内容格式不合法"
                    logger.warning(f"工具 '{tool_name}' 返回内容格式不合法: {plugin_output}")

                tool_summary_lines.append(msg)

            except Exception as e:
                msg = f"你调用了 {tool_name} 工具，执行异常：{e}"
                tool_summary_lines.append(msg)
                logger.error(msg)

        return "\n".join(tool_summary_lines)

    async def _call_llm(self, user_prompt: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        使用真实 LLM 生成工具调用 JSON 指令
        """
        tools_json = json.dumps(tools, ensure_ascii=False, indent=2)

        full_prompt = f"""
{SYSTEM_PROMPT}

用户输入:
{user_prompt}

可用工具:
{tools_json}

请输出 JSON：
"""

        response = await self.llm_request.generate_response_async(full_prompt)

        # 支持 tuple 返回
        if isinstance(response, tuple):
            response_str = response[0]
        else:
            response_str = response

        print(response_str)
        try:
            return json.loads(response_str)
        except Exception:
            logger.error(f"LLM 返回内容不是合法 JSON，将视为不调用工具。内容: {response_str}")
            return {}
