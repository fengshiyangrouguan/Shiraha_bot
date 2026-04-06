import json
from typing import List, Dict, Any, Optional

from src.common.logger import get_logger
from src.common.di.container import container

from src.llm_api.factory import LLMRequestFactory
from src.agent.world_model import WorldModel


logger = get_logger("motive")

class MotiveEngine:
    """
    动机引擎。
    负责根据世界状态和 Agent 的核心身份，生成高阶意图。
    """
    def __init__(self):
        """
        初始化动机引擎。
        """
        logger.info("开始初始化动机引擎...")
        
        llm_factory = container.resolve(LLMRequestFactory)
        self.llm_request = llm_factory.get_request(task_name="motive")
        
    def _format_capability_descriptions(self, collected_capability_descriptions: List[str]) -> str:
        """
        将收集到的内在驱动力描述格式化为字符串，供提示使用。
        """
        # 格式化内在驱动力描述为字符串
        if collected_capability_descriptions:
            formatted_capability = [f"- {desc}" for desc in collected_capability_descriptions]
            capability = "\n".join(formatted_capability)
            return capability
        else:
            capability = "无可用功能。"
            logger.warning("未收集到任何功能描述。")
            return capability
            


    def _build_prompt(self, world_model: WorldModel, capability: str) -> str:
        """
        根据 WorldModel 和 prompt_design.md 的设计构建 LLM 提示。
        """
        context = world_model.get_context_for_motive()
        time = world_model.get_current_time_string()
        
        prompt_template = f"""
你叫 {context['bot_name']}。
你是 {context['bot_identity']}。
你的性格是 {context['bot_personality']}，你的兴趣包括 {context['bot_interest']}。

{time}。
此刻你的心理状态是：{context['mood']}。

你最近一次活动的总结如下：
{context['action_summary']}

你目前可用软件和软件实时状态：
{capability}

## 你的自主性原则
1. **你可以选择“不作为”**：如果没有感兴趣的事，你可以发呆、反思或简单观察。
2. **你可以选择“改变焦点”**：如果当前任务让你疲惫，或者有更有趣的事吸引你，你可以产生切换注意力的欲望。
3. **动机并非指令**：你的动机应该是感性的内心独白（例如：“那本书好像更新了，想去看看”），而不是硬性的操作说明。

## 约束
- 严禁编造不存在的记忆或功能。
- 只输出一段自然的第一人称内心想法。

现在，基于上述全部信息，结合你的性格、情绪、兴趣和刚刚经历的事件，为自己生成一个最自然、最真实的人类式内在动机。
"""
        return prompt_template

    async def generate_motive(self) -> str:
        """
        生成一个高阶、模糊的动机。
        """
        
        world_model = container.resolve(WorldModel)
        capability = world_model.get_cortices_summaries()  # 获取各个 Cortex 的实时状态摘要，作为能力描述的一部分
        prompt_str = self._build_prompt(world_model,capability)
        print(f"生成动机的提示:\n{prompt_str}")

        try:
            response:str
            response, model_name = await self.llm_request.execute(
                prompt=prompt_str
            )
            
            if response:
                motive = response.strip().replace("\"", "").replace("'", "")
                logger.info(f"新的动机: {motive}")
                world_model.motive = motive
                return motive
            else:
                """ LLM 未返回内容，返回一个默认的休息动机 """
                logger.warning("LLM 未返回明确动机，选择休息一下。")
                world_model.motive = "休息一下"
                return "休息一下"

        except Exception as e:
            """ LLM 请求失败，返回一个处理错误的动机 """
            logger.error(f"生成动机时 LLM 请求失败: {e}")
            world_model.motive = "处理内部错误"
            return "处理内部错误"
        



# 之前的设计提示模板：
# {context['alert']}

# 你最近一次活动的总结如下：
# {context['action_summary']}

# 你目前拥有以下**可被激发的内在驱动力**：
# {impetus}

# 你的任务：
# 基于上述全部信息，结合你的性格、情绪、兴趣和刚刚经历的事件，为自己生成一个最自然、最真实的人类式内在动机。
# 以简洁、情绪化、带人类思考色彩的一句话表达你此刻最想做的事情。
# 语言风格更像你的内心自言自语，而不是程序化的指令。

# 你的动机可以是什么也不想干，选择休息或发呆也是可以的。

# 只输出一句自然的“内在意图表达”。