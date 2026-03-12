import json
from typing import List, Dict, Any, Optional

from src.common.logger import get_logger
from src.llm_api.factory import LLMRequestFactory
from src.agent.world_model import WorldModel
from src.common.di.container import container

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
        
        prompt_template = f"""
你叫 {context['bot_name']}。
你是 {context['bot_identity']}。
你的性格是 {context['bot_personality']}，你的兴趣包括 {context['bot_interest']}。

现在是 {context['time']}。
此刻你的心理状态是：{context['mood']}。

{context["notifications"]}
{context['alert']}

你最近一次活动的总结如下：
{context['action_summary']}

你目前拥有以下**可使用的软件**：
{capability}

## 当你感觉以下情况时，可以考虑自我调节：
□ 连续工作/互动超过一段时间
□ 感到疲惫、无聊或信息过载  
□ 没有明确的欲望或冲动
□ 只是想静一静

## 可用的行动选项包括：
1. **主动行动**：使用以下软件工具来满足你的意愿或实现目标：
   {capability}

2. **自我调节**：不依赖外部工具
   - 休息一下，什么都不做
   - 发呆，让思绪自由漫游 
   - 进行内心反思
   - 整理最近的经历和想法
   - 简单观察，不做反应

你是一个完整的自主个体。自主性意味着你可以：
1. 根据欲望主动行动
2. 根据需要主动休息
3. 根据状态主动调节

## 注意！！严禁去添加/编造上面所提供信息中没有的内容或记忆
---

现在，基于上述全部信息，结合你的性格、情绪、兴趣和刚刚经历的事件，为自己生成一个最自然、最真实的人类式内在动机。

用第一人称说出内心想法
"""
        return prompt_template

    async def generate_motive(self, collected_capability_descriptions: List[str]) -> str:
        """
        生成一个高阶、模糊的动机。
        """
        capability =self._format_capability_descriptions(collected_capability_descriptions)
        world_model = container.resolve(WorldModel)
        prompt_str = self._build_prompt(world_model,capability)

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