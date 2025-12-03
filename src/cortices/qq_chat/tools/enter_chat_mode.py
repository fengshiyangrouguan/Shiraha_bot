# src/cortices/qq_chat/tools/enter_chat_mode.py
from typing import Dict, Any, List

from src.cortices.tools_base import BaseTool
from src.cortices.manager import CortexManager
from src.agent.world_model import WorldModel
from src.common.logger import get_logger
from src.cortices.qq_chat.chat.qq_chat_data import QQChatData
from src.common.event_model.info_data import ConversationInfo

logger = get_logger("EnterChatModeTool")

class EnterChatModeTool(BaseTool):
    """
    当需要与某个QQ聊天对象进行复杂、多轮的对话时，进入此专属聊天模式。
    这个工具内部包含一个完整的 ReAct 循环，用于处理子任务。
    """

    def __init__(self, world_model: WorldModel, cortex_manager: CortexManager, adapter_id: str):
        self._world_model = world_model
        self._cortex_manager = cortex_manager
        self._adapter_id = adapter_id

    @property
    def scope(self) -> str:
        return "main"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "enter_qq_chat_mode",
            "description": "当需要在某个QQ聊天中进行复杂、多轮的对话时，进入此专属聊天模式。代理将专注于此聊天，直到任务完成或决定退出。",
            "parameters": {
                "conversation_id": {
                    "type": "string",
                    "description": "目标聊天（用户或群组）的ID。"
                },
                "task_description": {
                    "type": "string",
                    "description": "需要在此聊天中完成的具体任务描述。"
                }
            },
            "required_parameters": ["conversation_id", "task_description"]
        }

    async def execute(self, conversation_id: str, task_description: str) -> str:
        """
        执行专属聊天模式的 ReAct 循环。
        """
        logger.info(f"进入与 {conversation_id} 的专属聊天模式。任务: {task_description}")

        # 1. 获取上下文和子工具
        qq_chat_data: QQChatData = await self._world_model.get_cortex_data("qq_chat_data", QQChatData)
        chat_stream = qq_chat_data.get_or_create_stream(temp_convo_info)
        chat_history = chat_stream.build_chat_history_for_llm()

        sub_tools = self._cortex_manager.get_tool_schemas(scope="qq_deep_chat")
        
        # 3. 开始 ReAct 循环
        max_turns = 100 # 防止无限循环
        for turn in range(max_turns):
            logger.debug(f"专属聊天模式 Turn {turn+1}/{max_turns}")

            
            # b. 解析响应
            response_message = response.choices[0].message
            messages.append(response_message) # 将模型的回复加入历史

            if response_message.tool_calls:
                # c. 执行工具调用
                tool_call = response_message.tool_calls[0]
                logger.info(f"子规划器决定调用工具: {tool_call.func_name}({tool_call.args})")
                
                # 为子工具的 conversation_id 自动填充当前值
                if "conversation_id" not in tool_call.args:
                    tool_call.args["conversation_id"] = conversation_id
                
                tool_result = await self._cortex_manager.execute_tool(tool_call)
                logger.info(f"工具执行结果: {tool_result}")
                
                messages.append(Message(role="tool", tool_call_id=tool_call.id, name=tool_call.func_name, content=str(tool_result)))

            elif response_message.content:
                # d. LLM 决定任务完成
                final_answer = response_message.content
                logger.info(f"专属聊天模式结束。最终回复: {final_answer}")
                # 在退出前，也可以调用一个工具来发送最终消息
                chat_stream.mark_as_read() # 标记为已读
                await self._world_model.save_cortex_data("qq_chat_data", qq_chat_data)
                return f"与 {conversation_id} 的聊天任务已完成。总结: {final_answer}"

            else:
                logger.warning("LLM 未提供工具调用也未提供最终回复。")
                break
        
        chat_stream.mark_as_read() # 标记为已读
        await self._world_model.save_cortex_data("qq_chat_data", qq_chat_data)
        return f"与 {conversation_id} 的聊天任务已达到最大轮次，强制退出。"
